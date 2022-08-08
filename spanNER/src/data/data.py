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
        if self.cfg.task == 'scierc':
            gold_docs = [{
                **doc,
                'tokens': [[token for sent in doc['sentences'] for token in sent]],
                'ner': [ent for sent in doc['ner'] for ent in sent],
                'relations': [rel for sent in doc['relations'] for rel in sent if len(rel) > 0],
            }
                for doc in gold_docs]

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

    def sample_negatives(self, labels: List, sample_size: int):
        # assert sample_size < len(
        #     labels), f"sample size: {sample_size} large than population"
        # sample zeros indices
        negatives_indices = [idx for idx,
                             label in enumerate(labels) if label == 0]
        negative_samples = random.sample(negatives_indices, sample_size)
        return negative_samples

    def filter_samples(self, batch: List, negative_samples: List):
        entity_span = []
        labels = []
        span_mask = []
        global_idx_count = 0
        for batch_idx, ex in enumerate(batch):
            batch_spans = []
            batch_labels = []
            for span, label in zip(ex['spans'], ex['labels']):
                if label > 0:
                    batch_spans.append(span)
                    batch_labels.append(label)
                    span_mask.append(batch_idx)
                elif global_idx_count in negative_samples:
                    batch_spans.append(span)
                    batch_labels.append(0)
                    span_mask.append(batch_idx)
                global_idx_count += 1

            entity_span.append(batch_spans)
            labels += batch_labels

        labels = torch.tensor(labels)
        span_mask = torch.tensor(span_mask)
        return entity_span, labels, span_mask

    def collate_fn(self, batch):
        negative_samples = self.sample_negatives(
            [label for ex in batch for label in ex['labels']], sample_size=self.cfg.negative_samples_per_batch)

        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze(1)

        entity_span, labels, span_mask = self.filter_samples(
            batch, negative_samples)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": entity_span,
            "labels": labels,
            "span_mask": span_mask
        }


class ScirexDataset(Dataset):
    def __init__(self, cfg: Any, json_file: str, tokenizer: Any, entity_labels: List):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.entity_labels = entity_labels
        self.consolidated_dataset, self.global_labels = self._read(json_file)

    def _read(self, json_file: str) -> Tuple[List[Dict], List]:

        gold_docs = [json.loads(line) for line in open(json_file)]

        gold_docs = [{
            "doc_key": doc["doc_id"],
            # "sentences": doc["sentences"],
            'tokens': doc["words"],
            'ner': doc["ner"],
            'relations': doc["n_ary_relations"],
        } for doc in gold_docs]

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
                        for label in doc['ner'] if label[0] < self.cfg.max_length-2]

            span_classes = [self.entity_labels.index(
                label[-1]) for label in doc['ner'] if label[0] < self.cfg.max_length-2]

            spans = enumerate_spans(
                sentence=doc["tokens"][:self.cfg.max_length-2], max_span_width=self.cfg.max_span_length)

            for span in spans:
                candidate = [*span[:-1]]
                if candidate in span_ids:
                    class_pointer = span_ids.index(candidate)
                    class_idx = span_classes[class_pointer]
                    filtered_spans.append(span)
                    labels.append(class_idx)
                else:
                    filtered_spans.append(span)
                    labels.append(0)

            global_labels += labels

            docs_w_span_labels.append(
                {**doc, "spans": filtered_spans, "labels": labels})

        return docs_w_span_labels, global_labels

    def encode(self, docs: List[Dict]) -> List[Dict]:
        encoded_docs = []
        for doc in docs:
            text = self.tokenizer.convert_tokens_to_string(doc["tokens"])
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

    def sample_negatives(self, labels: List, sample_size: int):
        # assert sample_size < len(
        #     labels), f"sample size: {sample_size} large than population"
        # sample zeros indices
        negatives_indices = [idx for idx,
                             label in enumerate(labels) if label == 0]
        negative_samples = random.sample(negatives_indices, sample_size)
        return negative_samples

    def filter_samples(self, batch: List, negative_samples: List):
        entity_span = []
        labels = []
        span_mask = []
        global_idx_count = 0
        for batch_idx, ex in enumerate(batch):
            batch_spans = []
            batch_labels = []
            for span, label in zip(ex['spans'], ex['labels']):
                if label > 0:
                    batch_spans.append(span)
                    batch_labels.append(label)
                    span_mask.append(batch_idx)
                elif global_idx_count in negative_samples:
                    batch_spans.append(span)
                    batch_labels.append(0)
                    span_mask.append(batch_idx)
                global_idx_count += 1

            entity_span.append(batch_spans)
            labels += batch_labels

        labels = torch.tensor(labels)
        span_mask = torch.tensor(span_mask)
        return entity_span, labels, span_mask

    def collate_fn(self, batch):
        negative_samples = self.sample_negatives(
            [label for ex in batch for label in ex['labels']], sample_size=self.cfg.negative_samples_per_batch)

        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze(1)
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze(1)

        entity_span, labels, span_mask = self.filter_samples(
            batch, negative_samples)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": entity_span,
            "labels": labels,
            "span_mask": span_mask
        }
