from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import ast
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
from datasets import load_dataset
import ipdb
from tqdm import tqdm
import random
import logging
import logging
import time
from clearml import Task, StorageManager, Dataset as ClearML_Dataset

from common.data_structures import Dataset as EntityDataset
from common.const import task_ner_labels, get_labelmap
from common.utils import *
from models.utils import convert_dataset_to_samples, batchify, NpEncoder
from models.entity import EntityModel
import torch

Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("git+https://github.com/huggingface/datasets.git")
Task.add_requirements("hydra-core")
Task.add_requirements("pytorch-lightning")
Task.add_requirements("jsonlines")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def get_clearml_params(task: Task) -> Dict[str, Any]:
    """
    returns task params as a dictionary
    the values are casted in the required Python type
    """
    string_params = task.get_parameters_as_dict()
    clean_params = {}
    for k, v in string_params["General"].items():
        try:
            # ast.literal eval cannot read empty strings + actual strings
            # i.e. ast.literal_eval("True") -> True, ast.literal_eval("i am cute") -> error
            clean_params[k] = ast.literal_eval(v)
        except:
            # if exception is triggered, it's an actual string, or empty string
            clean_params[k] = v
    return OmegaConf.create(clean_params)


def get_dataset(split_name, cfg) -> DataLoader:
    """Get training and validation dataloaders"""
    clearml_data_object = ClearML_Dataset.get(
        dataset_name=cfg.clearml_dataset_name,
        dataset_project=cfg.clearml_dataset_project_name,
        dataset_tags=list(cfg.clearml_dataset_tags),
        # only_published=True,
    )
    dataset_path = clearml_data_object.get_local_copy()
    # raw_dataset = read_json(dataset_path+f"/{split_name}.jsonl")
    dataset = EntityDataset(dataset_path+f"/{split_name}.jsonl")
    return dataset

def calculate_loss_weights(spans_ner_label, num_ner_labels):
    weighted_ratio = torch.nn.init.constant_(torch.empty(num_ner_labels), 0.5)
    unique_class_distribution = torch.unique(spans_ner_label, return_counts=True)
    for idx, count in zip(unique_class_distribution[0],unique_class_distribution[1]):
        ratio = (count/spans_ner_label.size()[-1])*1
        weighted_ratio[idx] = 1-ratio
    return weighted_ratio

def train(cfg) -> Any:
    train_data = get_dataset("train", cfg)
    dev_data = get_dataset("dev", cfg)

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels["re3d"])

    train_samples, train_ner = convert_dataset_to_samples(
        train_data, max_span_length=cfg.max_span_length, ner_label2id=ner_label2id, context_window=cfg.context_window, negative_sample_ratio=cfg.negative_sample_ratio)
    train_batches = batchify(train_samples, batch_size=cfg.train_batch_size)

    train_labels = torch.tensor([labels for sample in train_samples for labels in sample['spans_label']])

    dev_samples, dev_ner = convert_dataset_to_samples(
        dev_data, max_span_length=cfg.max_span_length, ner_label2id=ner_label2id, context_window=cfg.context_window)
    dev_batches = batchify(dev_samples, batch_size=cfg.eval_batch_size)

    num_ner_labels = len(task_ner_labels["re3d"]) + 1
    loss_weights = calculate_loss_weights(train_labels, num_ner_labels)
    model = EntityModel(cfg, num_ner_labels=num_ner_labels, loss_weights=loss_weights)

    best_result = 0.0

    param_optimizer = list(model.bert_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if 'bert' in n]},
        {'params': [p for n, p in param_optimizer
                    if 'bert' not in n], 'lr': cfg.task_learning_rate}]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=cfg.learning_rate, correct_bias=not(cfg.bertadam))
    t_total = len(train_batches) * cfg.num_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(t_total*cfg.warmup_proportion), t_total)

    tr_loss = 0
    tr_examples = 0
    global_step = 0
    eval_step = len(train_batches) // cfg.eval_per_epoch

    for _ in tqdm(range(cfg.num_epoch)):
        if cfg.train_shuffle:
            random.shuffle(train_batches)
        for i in tqdm(range(len(train_batches))):
            output_dict = model.run_batch(train_batches[i], training=True)
            loss = output_dict['ner_loss']
            loss.backward()

            tr_loss += loss.item()
            tr_examples += len(train_batches[i])
            global_step += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % cfg.print_loss_step == 0:
                logger.info('Epoch=%d, iter=%d, loss=%.5f' %
                            (_, i, tr_loss / tr_examples))
                tr_loss = 0
                tr_examples = 0

            if global_step % eval_step == 0:
                f1, pred_ner = evaluate(model, dev_batches, dev_ner)
                if f1 > best_result:
                    best_result = f1
                    logger.info('!!! Best valid (epoch=%d): %.2f' %
                                (_, f1*100))
                    save_model(model, cfg)

    return model


def output_ner_predictions(model, ner_id2label, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                span_id = '%s::%d::(%d,%d)' % (
                    sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                if pred == 0:
                    continue
                ner_result[k].append(
                    [span[0]+off, span[1]+off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))


def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...' % (args.output_dir))
    model_to_save = model.bert_model.module if hasattr(
        model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1

    acc = l_cor / l_tot
    logger.info('Accuracy: %5f' % acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d' %
                (cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f' % (p, r, f1))
    logger.info('Used time: %f' % (time.time()-c_time))
    return f1, pred_ner


def test(cfg, model, task=None) -> None:
    test_data = get_dataset("test", cfg)
    ner_label2id, ner_id2label = get_labelmap(task_ner_labels["re3d"])

    test_samples, test_ner = convert_dataset_to_samples(
        test_data, max_span_length=cfg.max_span_length, ner_label2id=ner_label2id, context_window=cfg.context_window)
    test_batches = batchify(test_samples, batch_size=cfg.eval_batch_size)
    num_ner_labels = len(task_ner_labels["re3d"]) + 1

    evaluate(model, test_batches, test_ner)
    prediction_file = os.path.join(cfg.output_dir, cfg.test_pred_filename)
    output_ner_predictions(model, ner_id2label, test_batches, test_data,
                           output_file=prediction_file)
    if task:
        task.upload_artifact("preds", prediction_file)


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    tags = list(cfg.task_tags) + \
        ["debug"] if cfg.debug else list(cfg.task_tags)

    if cfg.do_train:
        task = Task.init(
            project_name="ER-extraction",
            task_name="pure-train",
            output_uri="s3://experiment-logging/storage/",
            tags=tags,
        )
    else:
        task = Task.init(
            project_name="ER-extraction",
            task_name="pure-predict",
            output_uri="s3://experiment-logging/storage/",
            tags=tags,
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    cfg = get_clearml_params(task)
    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.remote:
        task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
        task.execute_remotely(queue_name=cfg.queue, exit_process=True)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    if cfg.do_train:
        model = train(cfg)

    if cfg.do_eval:
        test(cfg, model, task)


if __name__ == "__main__":
    hydra_main()
