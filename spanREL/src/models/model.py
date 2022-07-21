from typing import Dict, Any, List, Tuple
from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig, AdamW, get_linear_schedule_with_warmup
from transformers.models.longformer.modeling_longformer import LongformerLMHead, _compute_global_attention_mask
import pytorch_lightning as pl
import torch.nn as nn
import torch
from common.utils import get_dataset, to_jsonl
from common.loss import FocalLoss
from sklearn.metrics import classification_report, precision_recall_fscore_support
from omegaconf import OmegaConf
import ipdb


class spanREL(pl.LightningModule):
    def __init__(self, args: OmegaConf, num_rel_labels: int, relation_loss_weights: torch.Tensor = None, task=None):
        super().__init__()
        self.args = args
        self.task = task
        self.num_ner_labels = num_rel_labels
        self.relation_loss_weights = relation_loss_weights
        self.span_width_embeddings = nn.Embedding(
            self.args.max_span_length+1, self.args.span_hidden_size)

        self.config = LongformerConfig.from_pretrained(
            'allenai/longformer-base-4096')
        self.config.gradient_checkpointing = True

        self.tokenizer = LongformerTokenizer.from_pretrained(
            'allenai/longformer-base-4096')
        self.tokenizer.model_max_length = self.args.max_length

        self.longformer = LongformerModel.from_pretrained(
            'allenai/longformer-base-4096', config=self.config)

        relation_classifier_input_dim = (self.config.hidden_size*2 +
                                         self.args.span_hidden_size)*2+self.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(relation_classifier_input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.hidden_size, self.num_ner_labels)
        )

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        global_attention_mask[:, :1] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def _get_span_embeddings(self, longformer_outputs, spans, span_mask):
        # ignore the CLS token in the outputs and offset
        unit_offset = self.args.max_length - 1
        sequence_output = longformer_outputs[0][:, 1:]

        # flatten the output to single batch
        flattened_batch = sequence_output.reshape(
            -1, self.config.hidden_size)

        flattened_spans = torch.tensor([(span[0]+idx*unit_offset, span[1]+idx*unit_offset, span[2])
                                        for idx, sample in enumerate(spans) for span in sample], device=self.device)

        span_starts = flattened_spans[:, 0]
        span_ends = flattened_spans[:, 1]
        span_widths = flattened_spans[:, 2].unsqueeze(0)

        # Get start embeddings
        start_span_embeddings = torch.index_select(
            flattened_batch, 0, span_starts)

        # Get end embeddings
        end_span_embeddings = torch.index_select(
            flattened_batch, 0, span_ends)

        # Get width embeddings
        span_width_embeddings = self.span_width_embeddings(span_widths)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (start_span_embeddings.unsqueeze(0), end_span_embeddings.unsqueeze(0), span_width_embeddings), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions
        span_mask = batch.pop("span_mask", None)
        span_subj = batch.pop("span_subj", None)
        span_obj = batch.pop("span_obj", None)
        labels = batch.pop("labels", None)

        outputs = self.longformer(
            **batch,
            global_attention_mask=self._set_global_attention_mask(batch["input_ids"]), output_hidden_states=True
        )

        cls_embeddings = outputs[0][:, :1].squeeze()[
            span_mask].unsqueeze(0)

        subj_embeddings = self._get_span_embeddings(
            outputs, span_subj, span_mask)

        obj_embeddings = self._get_span_embeddings(
            outputs, span_obj, span_mask)

        span_pair_embedding = torch.cat(
            (subj_embeddings, cls_embeddings, obj_embeddings), dim=-1)

        logits = self.classifier(span_pair_embedding).squeeze(0)

        if labels is not None:
            loss_fct = FocalLoss(gamma=3., reduction='sum')
            if self.relation_loss_weights is not None:
                loss_fct.weight = self.relation_loss_weights.to(self.device)
            rel_clf_loss = loss_fct(logits, labels)
            return (rel_clf_loss, logits)
        else:
            return (logits)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _ = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        self.log("train_loss", logs["train_loss"])

    def validation_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)

        if self.task:
            self.task.logger.report_scalar(
                title='val_loss', series='val', value=loss, iteration=batch_idx)
        preds = torch.argmax(logits, dim=-1)
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        val_preds = torch.cat([x["preds"] for x in outputs],
                              dim=0).view(-1).cpu().detach().tolist()

        val_labels = torch.cat([x["labels"] for x in outputs],
                               dim=0).view(-1).cpu().detach().tolist()

        logs = {
            "val_loss": val_loss_mean,
        }

        precision, recall, f1, support = precision_recall_fscore_support(
            val_labels, val_preds, average='macro')

        print(classification_report(val_labels, val_preds))

        self.log("val_loss", logs["val_loss"])
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

    def test_step(self, batch, batch_idx):
        span_mask = batch["span_mask"]
        loss, logits = self(**batch)
        preds = torch.argmax(logits, dim=-1)

        span_subj = batch["span_subj"]
        span_obj = batch["span_obj"]

        flattened_span_pairs = [[*span_subj, *span_obj] for sample_subj, sample_obj in zip(
            span_subj, span_obj) for span_subj, span_obj in zip(sample_subj, sample_obj)]

        span_pairs_w_preds = torch.tensor([[*span_pair, pred.cpu().detach().item()]
                                           for span_pair, pred in zip(flattened_span_pairs, preds)], device=self.device)

        reconstructed_preds = [span_pairs_w_preds[span_mask.eq(
            sample_idx)] for sample_idx in span_mask.unique()]

        return {"test_loss": loss, "preds": preds, "labels": batch["labels"], "reconstructed_preds": reconstructed_preds}

    def test_epoch_end(self, outputs):

        test_instance, _, relation_labels, _ = get_dataset(
            split_name='test', cfg=self.args)

        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()

        test_preds = torch.cat([x["preds"] for x in outputs],
                               dim=0).cpu().detach().tolist()

        test_labels = torch.cat([x["labels"] for x in outputs],
                                dim=0).view(-1).cpu().detach().tolist()

        reconstructed_preds = [[(*span_pair[:2], *span_pair[3:5], relation_labels[span_pair[-1]]) for span_pair in sample.cpu().detach().tolist() if span_pair[-1] is not 0]
                               for x in outputs for sample in x["reconstructed_preds"]]

        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average='macro')

        print(classification_report(test_labels, test_preds))

        preds = [{**{key: sample[key] for key in sample if key in ['doc_key', 'sentences', 'ner', 'relations', 'predicted_ner']}, "predicted_relations": pred}
                 for pred, sample in zip(reconstructed_preds, test_instance.consolidated_dataset)]

        to_jsonl("predictions.jsonl", preds)

        if self.task:
            self.task.upload_artifact("predictions", "predictions.jsonl")

        self.log("test_loss", test_loss_mean)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):

            if idx % 2 == 0:
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True

            # if idx<6:
            #     parameters.requires_grad=False
            # else:
            #     parameters.requires_grad=True

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.learning_rate)
        return optimizer
