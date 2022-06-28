task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    're3d': [
        "DocumentReference",
        "Location",
        "MilitaryPlatform",
        "Money",
        "Nationality",
        "Organisation",
        "Person",
        "Quantity",
        "Temporal",
        "Weapon",
        "Vehicle",
        "CommsIdentifier",
        "Coordinate",
        "Frequency",
    ]
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    're3d': [
        "Co-located",
        "Apart",
        "Belongs to",
        "In charge of",
        "Has the attribute of",
        "Is the same as",
        "Likes",
        "Dislikes",
        "Fighting against",
        "Military allies of",
        "Communicated with",
    ]
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
