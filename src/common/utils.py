import jsonlines
from collections import OrderedDict
import re
import json
import os
import ipdb
from typing import List, Dict, Any, Tuple


def to_jsonl(filename: str, file_obj):
    resultfile = open(filename, "wb")
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


def read_json(jsonfile):
    with open(jsonfile, "rb") as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object


def write_json(filename, file_object):
    with open(filename, "w") as file:
        file.write(json.dumps(file_object))


def read_json_multiple_templates(jsonfile):
    with open(jsonfile, "rb") as file:
        file_object = [json.loads(sample) for sample in file]

        for i in range(len(file_object)):
            if len(file_object[i]["templates"]) > 0:
                file_object[i]["templates"] = file_object[i]["templates"][0]
                del file_object[i]["templates"]["incident_type"]
            else:
                file_object[i]["templates"] = {
                    "Location": [],
                    "PerpInd": [],
                    "PerpOrg": [],
                    "PhysicalTarget": [],
                    "Weapon": [],
                    "HumTargetCivilian": [],
                    "HumTargetGovOfficial": [],
                    "HumTargetMilitary": [],
                    "HumTargetPoliticalFigure": [],
                    "HumTargetLegal": [],
                    "HumTargetOthers": [],
                    "KIASingle": [],
                    "KIAPlural": [],
                    "KIAMultiple": [],
                    "WIASingle": [],
                    "WIAPlural": [],
                    "WIAMultiple": [],
                }
    return file_object


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


