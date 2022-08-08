import json
import random
import torch
from torch.utils.data import Dataset
from typing import Callable, List, Set, Tuple, Dict, Any
import ipdb


def enumerate_spans(
    sentence: List,
    offset: int = 0,
    max_span_width: int = None,
    min_span_width: int = 1,
    filter_function: Callable[[List], bool] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.
    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.
    # Parameters
    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end, (end-start)))
    return spans


class EntityDataset(Dataset):
    def __init__(self, cfg: Any, json_file: str, tokenizer: Any, entity_labels: List):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.entity_labels = entity_labels
        self.consolidated_dataset, self.global_labels = self._read(json_file)

    def _read(self, json_file: str) -> Tuple[List[Dict], List]:
        gold_docs = [json.loads(line) for line in open(json_file)]
        # encoded_gold_docs = self.encode(gold_docs)
        gold_sent_w_span_labels, global_labels = self.get_spans(gold_docs)
        return gold_sent_w_span_labels, global_labels

    def get_spans(self, docs: List[Dict]) -> Tuple[List[Dict], List]:
        sent_w_span_labels = []
        global_labels = []
        for doc in docs:
            doc_key = doc['doc_key']
            offset = 0
            for sent, entities, relations in zip(doc['sentences'], doc['ner'], doc['relations']):
                labels = []
                filtered_spans = []

                span_ids = [[idx + offset for idx in label[:-1]]
                            for label in entities]  # [label[:-1] for label in entities]

                span_classes = [self.entity_labels.index(
                    label[-1]) for label in entities]

                spans = enumerate_spans(
                    sentence=sent[:self.cfg.max_length-2], max_span_width=self.cfg.max_span_length)

                for span in spans:
                    candidate = [*span[:-1]]
                    if candidate in span_ids:
                        class_pointer = span_ids.index(candidate)
                        class_idx = span_classes[class_pointer]
                        filtered_spans.append(span)
                        labels.append(class_idx)
                    elif random.random() < self.cfg.negative_sample_ratio:
                        filtered_spans.append(span)
                        labels.append(0)

                sent_w_span_labels.append(
                    {"doc_key": doc_key, "offset": offset, "sent": sent, "spans": filtered_spans, "labels": labels, "sent_length": len(sent)})

                offset += len(sent)
                global_labels += labels

        return sent_w_span_labels, global_labels

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        input_tokens = []
        input_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(input_tokens))
            sub_input_tokens = self.tokenizer.tokenize(token)
            input_tokens += sub_input_tokens
            end2idx.append(len(input_tokens)-1)
        input_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]]
                      for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list: List[Dict]):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = self.tokenizer.convert_tokens_to_string(sample['sent'])
            spans = sample['spans']
            spans_ner_label = sample['labels']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(
                tokens, spans, spans_ner_label)

            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)

            assert(bert_spans_tensor.shape[1] ==
                   spans_ner_label_tensor.shape[1])

            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length],
                                 self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full(
                    [1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat(
                    (attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]

            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full(
                    [1, spans_pad_length, bert_spans_tensor.size(2)], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full(
                    [1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat(
                    (spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat(
                    (spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat(
                    (final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat(
                    (final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat(
                    (final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat(
                    (final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat(
                    (final_spans_mask_tensor, spans_mask_tensor), dim=0)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):

        tokens_tensor, attention_mask_tensor, spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list=batch)

        return {
            "tokens_tensor": tokens_tensor,
            "attention_mask_tensor": attention_mask_tensor,
            "spans_tensor": spans_tensor,
            "spans_mask_tensor": spans_mask_tensor,
            "spans_ner_label_tensor": spans_ner_label_tensor,
            "sentence_length": sentence_length
        }

    # def collate_fn(self, batch):
    #     doc_keys = [ex['doc_key'] for ex in batch]
    #     input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
    #     attention_mask = torch.stack(
    #         [ex['attention_mask'] for ex in batch]).squeeze(1)
    #     entity_span = [ex['spans'] for ex in batch]
    #     entity_span = torch.tensor(
    #         [label for sample in entity_span for label in sample])
    #     labels = [ex['labels'] for ex in batch]
    #     labels = torch.tensor(
    #         [label for sample in labels for label in sample])
    #     span_mask = torch.tensor([idx for idx, ex in enumerate(
    #         batch) for span in ex['spans']])

    #     return {
    #         "doc_keys": doc_keys,
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "spans": entity_span,
    #         "labels": labels,
    #         "span_mask": span_mask
    #     }
