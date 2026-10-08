"""Fixed-model replay and exploratory joint census on public PSL annotations.

No tuning or new model configuration. Export aggregates, coded AP predictions,
and at most eight prospectively rule-selected public lexical illustrations.
Protocol: research/terminal_control_explanation_protocol_r1.md.
"""


from __future__ import annotations


import argparse


from collections import Counter, defaultdict


from datetime import datetime, timezone


import hashlib


from itertools import combinations


import json


from pathlib import Path


import sys


import time


import numpy as np


import scipy


import model as sp


import terminal as tc


EXAMPLE_NAMESPACE = 'project8-terminal-explanation-examples-r1\0'


OUTCOMES = ['both_correct', 'only_all_position_correct',
            'only_terminal5_correct', 'neither_correct']


DIMENSIONS = ['ending_group', 'recorded_syllable_count',
              'longest_training_terminal_feature_letters', 'character_length']


ENDINGS = [*tc.ENDING_SETS, 'remainder_not_morphologically_classified']


MODELS = ['all_marks_removed__all_position',
          'all_marks_removed__terminal_tagged_1to5']


COUNT_FIELDS = ['observations', 'both_correct', 'only_a_correct', 'only_b_correct',
                'both_wrong', 'a_correct', 'b_correct', 'net_correct_a_minus_b']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(obj):
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def counts(values):
    c = Counter(values)
    result = {k: int(c[k]) for k in OUTCOMES}
    result['observations'] = sum(result.values())
    result['all_position_correct'] = result['both_correct'] + result['only_all_position_correct']
    result['terminal5_correct'] = result['both_correct'] + result['only_terminal5_correct']
    result['net_correct_all_minus_terminal5'] = result['only_all_position_correct'] - result['only_terminal5_correct']
    return result


def four_way(a, b):
    return OUTCOMES[0 if a == 2 and b == 2 else 1 if a == 2 else 2 if b == 2 else 3]


def joint_table(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in DIMENSIONS)].append(row['paired_correctness'])
    return [{**dict(zip(DIMENSIONS, key)), **counts(values)}
            for key, values in sorted(buckets.items())]


def project_table(joint, keys, universe=None):
    buckets = defaultdict(Counter)
    for cell in joint:
        key = tuple(cell[k] for k in keys)
        for label in OUTCOMES:
            buckets[key][label] += cell[label]
    if universe is not None:
        assert len(keys) == 1
        for value in universe:
            buckets[(value,)]
    cells = []
    for key, c in sorted(buckets.items()):
        # counts accepts an iterable; elements() preserves repeated counts.
        cells.append({**dict(zip(keys, key)), **counts(c.elements())})
    return cells


def projected_tables(joint, universe):
    one = {d: project_table(joint, [d], universe[d]) for d in DIMENSIONS}
    two = {'__'.join(pair): project_table(joint, list(pair))
           for pair in combinations(DIMENSIONS, 2)}
    covered = [{**c, 'five_letter_coverage': '5' if c[DIMENSIONS[2]] == 5 else '<5'}
               for c in joint]
    binary = {d: project_table(covered, [d, 'five_letter_coverage'])
              for d in (DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[3])}
    return one, two, binary


def selected_examples(rows, test_words):
    out = []
    for outcome in OUTCOMES:
        for coverage in ['<5', '5']:
            candidates = [r for r in rows if r['paired_correctness'] == outcome
                          and ('5' if r[DIMENSIONS[2]] == 5 else '<5') == coverage]
            cell = {'paired_correctness': outcome, 'five_letter_coverage': coverage,
                    'eligible_types_in_selection_cell': len(candidates)}
            if not candidates:
                cell['example'] = None
            else:
                def rank(row):
                    word = test_words[row['test_index']]
                    return hashlib.sha256((EXAMPLE_NAMESPACE + word).encode()).hexdigest(), word
                chosen = min(candidates, key=rank)
                word = test_words[chosen['test_index']]
                cell['example'] = {
                    'test_index': chosen['test_index'], 'canonical_public_spelling': word,
                    'fully_folded_model_input': sp.stripped(word), 'recorded_label': 'antepenult',
                    **{k: chosen[k] for k in DIMENSIONS},
                    'all_position_prediction': tc.CLASS_NAMES[chosen['all_position_class']],
                    'terminal5_prediction': tc.CLASS_NAMES[chosen['terminal5_class']],
                    'selection_sha256': rank(chosen)[0]}
            out.append(cell)
    return out


def checks():
    assert len(sp.stripped('A\u0301çã')) == 3
    assert [four_way(*x) for x in [(2, 2), (2, 0), (1, 2), (0, 1)]] == OUTCOMES
    samples = []
    for i, outcome in enumerate(OUTCOMES):
        samples.append({'test_index': i, 'ending_group': ENDINGS[0] if i < 2 else ENDINGS[-1],
                        'recorded_syllable_count': '3' if i % 2 == 0 else 'multiple_recorded:3|4',
                        'longest_training_terminal_feature_letters': 5 if i < 2 else 4,
                        'character_length': 8, 'paired_correctness': outcome,
                        'all_position_class': [2, 2, 1, 0][i], 'terminal5_class': [2, 0, 2, 1][i]})
    joint = joint_table(samples)
    coverage = project_table(joint, [DIMENSIONS[2]])
    assert coverage[0] == {DIMENSIONS[2]: 4, **counts(OUTCOMES[2:])}
    assert coverage[1] == {DIMENSIONS[2]: 5, **counts(OUTCOMES[:2])}
    zero = project_table(joint, ['ending_group'], ENDINGS)
    assert sum(c['observations'] for c in zero) == 4
    assert sum(c['observations'] == 0 for c in zero) == 2
    words = ['synthetic_a', 'synthetic_b', 'synthetic_c', 'synthetic_d']
    ex1, ex2 = selected_examples(samples, words), selected_examples(list(reversed(samples)), words)
    assert ex1 == ex2 and sum(c['example'] is not None for c in ex1) == 4
    return {'status': 'PASS', 'fixtures': 'explicit synthetic strings, source-count sets and outcomes',
            'checks': ['combining marks and character length', 'four paired outcome groups',
                       'exact joint/marginal counts', 'explicit absent ending categories',
                       'multiple recorded syllable-count category', 'order-invariant example rule'],
            'existing_feature_checks': tc.checks()}

