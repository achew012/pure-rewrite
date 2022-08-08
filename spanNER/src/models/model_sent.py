from typing import Dict, Any, List, Tuple
# from transformers import LongformerForMaskedLM, LongformerModel, LongformerTokenizer, LongformerConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
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
            nn.Linear(self.config.hidden_size*2 +
                      self.args.span_hidden_size, self.config.hidden_size),
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

    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def _get_span_embeddings(self, input_ids: torch.Tensor, spans: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask)

        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        # sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = self.batched_index_select(
            sequence_output, 1, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = self.batched_index_select(
            sequence_output, 1, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.span_width_embeddings(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions
        input_ids, attention_mask, spans, spans_mask, = batch['tokens_tensor'], batch[
            'attention_mask_tensor'], batch['spans_tensor'], batch['spans_mask_tensor']
        spans_ner_label = batch.pop('spans_ner_label_tensor', None)

        spans_embedding = self._get_span_embeddings(
            input_ids, spans, attention_mask=attention_mask)
        logits = self.classifier(spans_embedding)

        if spans_ner_label is not None:
            loss_fct = FocalLoss(
                weight=self.entity_loss_weights.to(self.device), gamma=3., reduction='sum')
            # loss_fct = nn.CrossEntropyLoss(
            #     weight=self.entity_loss_weights.to(self.device), reduction='sum')

            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                # active_labels = torch.where(
                #     active_loss, spans_ner_label.view(-1), torch.tensor(
                #         loss_fct.ignore_index).type_as(spans_ner_label)
                # )
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(
                        -100).type_as(spans_ner_label)
                )

                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss, _, _ = self(**batch)
        loss = loss.sum()
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
        loss, logits, _ = self(**batch)

        if self.task:
            self.task.logger.report_scalar(
                title='val_loss', series='val', value=loss, iteration=batch_idx)
        preds = torch.argmax(logits, dim=2)
        return {"val_loss": loss, "preds": preds, "labels": batch["spans_ner_label_tensor"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack(
            [x["val_loss"] for x in outputs]).mean()

        val_preds = torch.cat([x["preds"].view(-1) for x in outputs],
                              dim=0).cpu().detach().tolist()

        val_labels = torch.cat([x["labels"].view(-1) for x in outputs],
                               dim=0).cpu().detach().tolist()

        logs = {
            "val_loss": val_loss_mean,
        }

        precision, recall, f1, support = precision_recall_fscore_support(
            val_labels, val_preds, average='macro')

        print(classification_report(val_labels, val_preds))
        print(
            f"loss: {val_loss_mean}, precision: {precision}, recall: {recall}, f1: {f1}")

        self.log("val_loss", logs["val_loss"])
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

    def test_step(self, batch, batch_idx):
        loss, logits, _ = self(**batch)
        _, predicted_labels = torch.argmax(logits, dim=2)

        predicted_labels = predicted_labels.cpu().numpy()

        return {"test_loss": loss, "preds": predicted_labels}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack(
            [x["test_loss"] for x in outputs]).mean()

        test_preds = torch.cat([x["preds"].view(-1) for x in outputs],
                               dim=0).cpu().detach().tolist()

        test_labels = torch.cat([x["labels"].view(-1) for x in outputs],
                                dim=0).cpu().detach().tolist()

        logs = {
            "test_loss": test_loss_mean,
        }

        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average='macro')

        print(classification_report(test_labels, test_preds))

        self.log("test_loss", logs["test_loss"])
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)

        # test_instance, entity_labels, _, _ = get_dataset(
        #     split_name='test', cfg=self.args)

        # test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()

        # test_preds = torch.cat([x["preds"] for x in outputs],
        #                        dim=0).cpu().detach().tolist()

        # test_labels = torch.cat([x["labels"] for x in outputs],
        #                         dim=0).view(-1).cpu().detach().tolist()

        # reconstructed_preds = [[(span[0], span[1], entity_labels[span[2]]) for span in sample.cpu().detach().tolist() if span[2] != 0]
        #                        for x in outputs for sample in x["reconstructed_preds"]]
        # precision, recall, f1, support = precision_recall_fscore_support(
        #     test_labels, test_preds, average='macro')

        # print(classification_report(test_labels, test_preds))

        # preds = [{**{key: sample[key] for key in sample if key in ['doc_key', 'sentences', 'ner', 'relations', 'predicted_ner']}, "predicted_ner": pred}
        #          for pred, sample in zip(reconstructed_preds, test_instance.consolidated_dataset)]

        # to_jsonl("predictions.jsonl", preds)

        # if self.task:
        #     self.task.upload_artifact("predictions", "predictions.jsonl")

        # self.log("test_loss", test_loss_mean)
        # self.log("test_precision", precision)
        # self.log("test_recall", recall)
        # self.log("test_f1", f1)

    # Freeze weights?
    def configure_optimizers(self):
        # Freeze alternate layers of longformer
        # for idx, (name, parameters) in enumerate(self.model.named_parameters()):
        #     if idx % 2 == 0:
        #         parameters.requires_grad = False
        #     else:
        #         parameters.requires_grad = True

        # weight_decay = 1e-4
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)

        param_optimizer = list(self.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                        if 'bert' not in n], 'lr': self.args.learning_rate*10}]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=30, gamma=0.1, verbose=True)
        # return (
        #     {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val_loss",
        #         },
        #     },
        # )
        return optimizer
