from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import os
import ast
import logging
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
import ipdb
import json
from models.model import spanNER
from common.utils import get_dataset
from clearml import Task


Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")
Task.add_requirements("git+https://github.com/huggingface/datasets.git")
# Task.add_requirements("hydra-core")
# Task.add_requirements("pytorch-lightning")
# Task.add_requirements("jsonlines")

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


def get_dataloader(split_name: str, cfg) -> Tuple[DataLoader, List, List]:
    data_instance, entity_labels, relation_labels, loss_weights = get_dataset(split_name, cfg)

    if split_name == "train":
        return DataLoader(
            data_instance, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=data_instance.collate_fn), entity_labels, relation_labels, loss_weights
    else:
        return DataLoader(
            data_instance, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=data_instance.collate_fn), entity_labels, relation_labels


def train(cfg) -> Any:
    train_loader, entity_labels, relation_labels, entity_loss_weights = get_dataloader(
        "train", cfg)
    val_loader, _, _ = get_dataloader("dev", cfg)
    entity_loss_weights = None
    # print(f"Loss weights: {entity_loss_weights}")
    model = spanNER(cfg, num_ner_labels=len(entity_labels),
                    entity_loss_weights=entity_loss_weights)

    callbacks = []

    if cfg.checkpointing:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="./",
            filename="best_ner",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            every_n_epochs=10,
        )
        callbacks.append(checkpoint_callback)

    if cfg.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=12, verbose=True, mode="min")
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        gpus=cfg.gpu, max_epochs=cfg.num_epoch, callbacks=callbacks, check_val_every_n_epoch=cfg.eval_per_epoch, enable_checkpointing=cfg.checkpointing)
    trainer.fit(model, train_loader, val_loader)

    return model


def evaluate(cfg, model) -> None:
    test_loader, _, _ = get_dataloader("test", cfg)
    trainer = pl.Trainer(gpus=cfg.gpu)
    trainer.test(model, test_loader)


@ hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    tags = list(cfg.task_tags) + \
        ["debug"] if cfg.debug else list(cfg.task_tags)

    if cfg.do_train:
        task = Task.init(
            project_name="ER-extraction",
            task_name="spanNER-train",
            output_uri="s3://experiment-logging/storage/",
            tags=tags,
        )
    else:
        task = Task.init(
            project_name="ER-extraction",
            task_name="spanNER-predict",
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

    if cfg.do_train:
        model = train(cfg)

    if cfg.do_eval and model:
        if cfg.trained_model_path:
            checkpoint_path = cfg.trained_model_path
        elif len(task.models['output']) > 0:
            checkpoint_path = task.models['output'][0].get_weights()
        else:
            print(f"trained outputs: {task.models['output']}")
            checkpoint_path = None

        if checkpoint_path:
            if cfg.task == "scirex":
                num_ner_labels = 5
            elif cfg.task == "scierc":
                num_ner_labels = 7
            elif cfg.task == "re3d":
                num_ner_labels = 15

            model = model.load_from_checkpoint(
                checkpoint_path, args=cfg, num_ner_labels=num_ner_labels)
            evaluate(cfg, model)
            task.upload_artifact('predictions', './predictions.jsonl')
        else:
            print("No checkpoint path found. Skipping evaluation")


if __name__ == "__main__":
    hydra_main()
