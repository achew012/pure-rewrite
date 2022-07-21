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
        if self.cfg.debug:
            gold_docs = [json.loads(line) for idx, line in enumerate(
                open(json_file)) if idx < 50]
        else:
            gold_docs = [json.loads(line) for line in open(json_file)]
        encoded_gold_docs = self.encode(gold_docs)
        encoded_gold_docs_w_span_labels, global_labels = self.get_spans(
            encoded_gold_docs)
        return encoded_gold_docs_w_span_labels, global_labels

    def get_spans(self, docs: List[Dict]) -> Tuple[List[Dict], List]:
        docs_w_span_labels = []
        global_labels = []
        for doc in docs:
            labels = []
            filtered_spans = []
            span_ids = [label[:-1]
                        for label in doc['ner']]

            span_classes = [self.entity_labels.index(
                label[-1]) for label in doc['ner']]

            spans = enumerate_spans(
                sentence=doc["tokens"][0][:self.cfg.max_length-2], max_span_width=self.cfg.max_span_length)

            for span in spans:
                candidate = [*span[:-1]]
                if candidate in span_ids:
                    class_pointer = span_ids.index(candidate)
                    class_idx = span_classes[class_pointer]
                    filtered_spans.append(span)
                    labels.append(class_idx)
                else:
                    if random.random() < self.cfg.negative_sample_ratio:
                        filtered_spans.append(span)
                        labels.append(0)

            global_labels += labels

            docs_w_span_labels.append(
                {**doc, "spans": filtered_spans, "labels": labels})

        return docs_w_span_labels, global_labels

    def encode(self, docs: List[Dict]) -> List[Dict]:
        encoded_docs = []
        for doc in docs:
            text = self.tokenizer.convert_tokens_to_string(doc["tokens"][0])
            encodings = self.tokenizer(
                text, padding="max_length", truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
            encoded_docs.append(
                {**doc, "input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]})
        return encoded_docs

    def __len__(self):
        return len(self.consolidated_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.consolidated_dataset[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze(1)
        entity_span = [ex['spans'] for ex in batch]
        labels = [ex['labels'] for ex in batch]
        labels = torch.tensor(
            [label for sample in labels for label in sample])
        span_mask = torch.tensor([idx for idx, ex in enumerate(
            batch) for span in ex['spans']])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": entity_span,
            "labels": labels,
            "span_mask": span_mask
        }
