from typing import Dict, Any, List, Tuple
# from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
# from transformers.models.longformer.modeling_longformer import LongformerLMHead, _compute_global_attention_mask
import pytorch_lightning as pl
import torch.nn as nn
import torch
from common.utils import get_dataset, to_jsonl
from common.loss import FocalLoss
from sklearn.metrics import classification_report, precision_recall_fscore_support
from omegaconf import OmegaConf
import ipdb


class spanNER(pl.LightningModule):
    def __init__(self, args: OmegaConf, num_ner_labels: int, entity_loss_weights: torch.Tensor = None, task=None):
        super().__init__()
        self.args = args
        self.task = task
        self.num_ner_labels = num_ner_labels
        self.entity_loss_weights = entity_loss_weights
        self.span_width_embeddings = nn.Embedding(
            self.args.max_span_length+1, self.args.span_hidden_size)

        self.config = AutoConfig.from_pretrained(
            self.args.model)
        self.config.gradient_checkpointing = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model)
        self.tokenizer.model_max_length = self.args.max_length

        self.model = AutoModel.from_pretrained(
            self.args.model, config=self.config)

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size*3 +
                      self.args.span_hidden_size, self.config.hidden_size),
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

    def _get_span_embeddings(self, outputs, spans, span_mask):

        # ignore the CLS token in the outputs and offset
        unit_offset = self.args.max_length - 1
        cls_embeddings = outputs[0][:, :1].squeeze()[
            span_mask].unsqueeze(0)

        sequence_output = outputs[0][:, 1:]

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
            (start_span_embeddings.unsqueeze(0), end_span_embeddings.unsqueeze(0), span_width_embeddings, cls_embeddings), dim=-1)

        # spans_embedding = torch.cat(
        #     (start_span_embeddings.unsqueeze(0), end_span_embeddings.unsqueeze(0), span_width_embeddings), dim=-1)

        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def calculate_loss_weights(self, labels: torch.Tensor, num_labels: int) -> torch.Tensor:
        weighted_ratio = torch.nn.init.constant_(torch.empty(num_labels), 0.9)
        unique_class_distribution = torch.unique(
            labels, return_counts=True)
        for idx, count in zip(unique_class_distribution[0], unique_class_distribution[1]):
            ratio = (count/labels.size()[-1])
            weighted_ratio[idx] = 1-ratio
        return weighted_ratio

    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions
        span_mask = batch.pop("span_mask", None)
        spans = batch.pop("spans", None)
        labels = batch.pop("labels", None)

        outputs = self.model(
            **batch,
            output_hidden_states=True,
            # global_attention_mask=self._set_global_attention_mask(batch["input_ids"])
        )

        span_embeddings = self._get_span_embeddings(outputs, spans, span_mask)
        logits = self.classifier(span_embeddings).squeeze(0)

        if labels is not None:
            entity_loss_weights = self.calculate_loss_weights(
                labels, self.num_ner_labels)
            loss_fct = FocalLoss(
                weight=entity_loss_weights.to(self.device), gamma=3., reduction='sum')
            # loss_fct = torch.nn.CrossEntropyLoss(
            #     weight=entity_loss_weights.to(self.device), reduction='sum')

            span_clf_loss = loss_fct(logits, labels)
            return (span_clf_loss, logits)
        else:
            return (logits)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        doc_keys = batch.pop("doc_keys", None)
        loss, _ = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack(
            [x["loss"] for x in outputs]).mean()

        logs = {
            "train_loss": train_loss_mean,
        }

        self.log("train_loss", logs["train_loss"])

    def validation_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels = batch
        doc_keys = batch.pop("doc_keys", None)
        loss, logits = self(**batch)

        if self.task:
            self.task.logger.report_scalar(
                title='val_loss', series='val', value=loss, iteration=batch_idx)
        preds = torch.argmax(logits, dim=-1)
        return {"val_loss": loss, "preds": preds, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack(
            [x["val_loss"] for x in outputs]).mean()

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
        doc_keys = batch.pop("doc_keys", None)
        span_mask = batch["span_mask"]
        loss, logits = self(**batch)
        preds = torch.argmax(logits, dim=-1)
        spans = batch["spans"]

        flattened_spans = [span for sample in spans for span in sample]
        spans_w_preds = torch.tensor([(span[0], span[1], pred.cpu().detach().item())
                                      for span, pred in zip(flattened_spans, preds)], device=self.device)

        reconstructed_preds = [spans_w_preds[span_mask.eq(
            sample_idx)] for sample_idx in span_mask.unique()]

        return {"test_loss": loss, "preds": preds, "labels": batch["labels"], "reconstructed_preds": reconstructed_preds}

    def test_epoch_end(self, outputs):

        test_instance, entity_labels, _, _ = get_dataset(
            split_name='test', cfg=self.args)

        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()

        test_preds = torch.cat([x["preds"] for x in outputs],
                               dim=0).cpu().detach().tolist()

        test_labels = torch.cat([x["labels"] for x in outputs],
                                dim=0).view(-1).cpu().detach().tolist()

        reconstructed_preds = [[(span[0], span[1], entity_labels[span[2]]) for span in sample.cpu().detach().tolist() if span[2] != 0]
                               for x in outputs for sample in x["reconstructed_preds"]]
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average='macro')

        print(classification_report(test_labels, test_preds))

        preds = [{**{key: sample[key] for key in sample if key in ['doc_key', 'sentences', 'ner', 'relations', 'predicted_ner']}, "predicted_ner": pred}
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
        # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
        #     if idx % 2 == 0:
        #         parameters.requires_grad = False
        #     else:
        #         parameters.requires_grad = True

        weight_decay = 1e-4

        param_optimizer = list(self.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                        if 'bert' not in n], 'lr': self.args.learning_rate*10}]

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
        #                               lr=self.args.learning_rate, weight_decay=weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=10, gamma=0.2, verbose=True)

        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=self.args.learning_rate, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=self.args.learning_rate*0.1, max_lr=self.args.learning_rate*10, verbose=True)

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )
