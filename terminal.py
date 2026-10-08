"""Exploratory terminal-window controls on the public PSL; aggregate export only.

Protocol: research/terminal_control_protocol_r1.md and its prospective r1.1
amendment. Reuses audited generated fitting/normalization utilities, never
protected source code. No words, vocabulary or fitted parameters are exported.
"""


from __future__ import annotations


import argparse


from collections import Counter, defaultdict


import csv


from datetime import datetime, timezone


import hashlib


import json


from pathlib import Path


import sys


import time


import numpy as np


import scipy


from scipy import sparse


import model as sp


import marks as sn


GRID = [0.1, 1.0, 10.0, 100.0]


FAMILIES = ['all_position', 'terminal_tagged_1to5', 'terminal_subset_1to4']


REPRESENTATIONS = {'all_marks_removed': sp.stripped,
                   'acute_circumflex_removed': sn.selective}


ENDING_SETS = {
    'ends_ico_ica_icos_icas': ('ico', 'ica', 'icos', 'icas'),
    'ends_logo_loga_logos_logas': ('logo', 'loga', 'logos', 'logas'),
    'ends_oso_osa_osos_osas': ('oso', 'osa', 'osos', 'osas'),
}


CLASS_NAMES = ['final', 'penult', 'antepenult']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def features(word, family):
    if family == 'all_position':
        return sp.grams(word)
    if family == 'terminal_subset_1to4':
        return {g for g in sp.grams(word) if g.endswith('$')}
    if family == 'terminal_tagged_1to5':
        return {f'S{k}:' + word[-k:] for k in range(1, min(5, len(word)) + 1)}
    raise ValueError('Unknown feature family')


def vocabulary(words, family):
    if family == 'all_position':
        return sp.vectorizer(words)
    if family == 'terminal_subset_1to4':
        parent = sp.vectorizer(words)
        return {g: i for i, g in enumerate(sorted(g for g in parent if g.endswith('$')))}
    counts = Counter(g for word in words for g in features(word, family))
    return {g: i for i, g in enumerate(sorted(g for g, n in counts.items() if n >= 2))}


def matrix(words, vocab, family):
    if family in ('all_position', 'terminal_subset_1to4'):
        # sp.matrix filters to this vocabulary before calculating its own norm.
        return sp.matrix(words, vocab)
    indptr, indices, data = [0], [], []
    for word in words:
        ix = sorted(vocab[g] for g in features(word, family) if g in vocab)
        indices.extend(ix)
        data.extend([1.0 / np.sqrt(len(ix))] * len(ix) if ix else [])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.array(data, dtype=float), np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int32)), shape=(len(words), len(vocab)))


def matrix_summary(X):
    counts = np.diff(X.indptr)
    return {'observations': X.shape[0], 'feature_dimension': X.shape[1],
            'nonzero_entries': int(X.nnz), 'zero_vectors': int(np.sum(counts == 0)),
            'active_features_min': int(counts.min()) if len(counts) else None,
            'active_features_max': int(counts.max()) if len(counts) else None,
            'active_features_mean': float(counts.mean()) if len(counts) else None}


def metrics(labels, prob):
    record = sp.metrics(labels, prob)
    record['macro_recall'] = float(np.mean(record['recall_by_class']))
    record['macro_f1_definition'] = 'mean_c[2 TP_c / (true_support_c + predicted_support_c)]'
    record['class_universe'] = CLASS_NAMES
    record['zero_denominator_policy'] = 'zero, as in original audited metrics'
    return record


def paired_counts(labels, a, b, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    labels, a, b = labels[mask], a[mask], b[mask]
    if not len(labels):
        return {'observations': 0, 'difference': None}
    ca, cb = a.argmax(axis=1) == labels, b.argmax(axis=1) == labels
    both = int(np.sum(ca & cb))
    only_a, only_b = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    neither = int(np.sum(~ca & ~cb))
    assert both + only_a + only_b + neither == len(labels)
    net = only_a - only_b
    assert net == int(ca.sum()) - int(cb.sum())
    return {'observations': len(labels), 'both_correct': both,
            'only_a_correct': only_a, 'only_b_correct': only_b,
            'both_wrong': neither, 'a_correct': int(ca.sum()),
            'b_correct': int(cb.sum()), 'net_correct_a_minus_b': net,
            'difference': net / len(labels),
            'a_log_loss': float(-np.log(np.maximum(a[np.arange(len(labels)), labels], 1e-300)).mean()),
            'b_log_loss': float(-np.log(np.maximum(b[np.arange(len(labels)), labels], 1e-300)).mean())}


def contrast(labels, a, b):
    return {'all_classes': paired_counts(labels, a, b),
            'by_recorded_class': {c: paired_counts(labels, a, b, labels == i)
                                  for i, c in enumerate(CLASS_NAMES)}}


def ending_stratum(word):
    folded = sp.stripped(word)
    hits = [name for name, endings in ENDING_SETS.items() if folded.endswith(endings)]
    if len(hits) > 1:
        raise ValueError('Ending sets unexpectedly overlap')
    return hits[0] if hits else 'remainder_not_morphologically_classified'


def longest_terminal(word, vocab, family):
    if family == 'terminal_tagged_1to5':
        return max((int(g[1:g.index(':')]) for g in features(word, family) if g in vocab), default=0)
    if family == 'terminal_subset_1to4':
        return max((len(g.removeprefix('^').removesuffix('$'))
                    for g in features(word, family) if g in vocab), default=0)
    raise ValueError('Longest-terminal stratum requires a terminal family')


def syllable_strata(path, eligible_words):
    eligible = set(eligible_words)
    counts = defaultdict(set)
    with path.open() as f:
        for row in csv.DictReader(f):
            word = sp.canonical(row['word'])
            if word in eligible:
                # Preserve the source's recorded count set, including variants.
                counts[word].add(int(row['nSyl']))
    assert set(counts) == eligible
    return {w: (str(next(iter(ns))) if len(ns) == 1 else
                'multiple_recorded:' + '|'.join(map(str, sorted(ns))))
            for w, ns in counts.items()}


def stratified(words, labels, a, b, transformed, terminal_vocab, terminal_family, syllables):
    dimensions = {
        'fold_changes_spelling': ['changed' if sp.stripped(w) != w else 'unchanged' for w in words],
        'recorded_syllable_count': [syllables[w] for w in words],
        'longest_training_terminal_feature_letters':
            [str(longest_terminal(w, terminal_vocab, terminal_family)) for w in transformed],
        'predeclared_string_ending': [ending_stratum(w) for w in words],
    }
    result = {}
    for name, values in dimensions.items():
        bins = defaultdict(list)
        for i, value in enumerate(values):
            bins[(int(labels[i]), value)].append(i)
        cells = []
        for (label, value), indices in sorted(bins.items()):
            ix = np.array(indices, dtype=np.int64)
            cells.append({'recorded_class': CLASS_NAMES[label], 'stratum': value,
                          **paired_counts(labels[ix], a[ix], b[ix])})
        assert sum(c['observations'] for c in cells) == len(words)
        result[name] = cells
    return {'terminal_stratum_vocabulary': terminal_family,
            'interpretation': 'orthographic/source-field strata, not morphology or new human labels',
            'dimensions_crossed_separately_with_class': result}


def checks():
    # Every word here is an explicit synthetic fixture, not a lexical export.
    words = ['aaaaxabcd', 'aaabxabcd', 'bbbbxabcd', 'bbbcxabcd', 'cccdyabcd', 'ccceyabcd']
    vocabs = {f: vocabulary(words, f) for f in FAMILIES}
    assert set(vocabs['terminal_subset_1to4']) == {
        g for g in vocabs['all_position'] if g.endswith('$')}
    for family in FAMILIES:
        X = matrix(words + ['zzzzzz'], vocabs[family], family)
        norm = np.array(X.multiply(X).sum(axis=1)).ravel()
        np.testing.assert_allclose(norm, [1] * len(words) + [0], atol=1e-14)
        assert X[-1].nnz == 0
    for family in FAMILIES[1:]:
        X = matrix([words[0], words[2]], vocabs[family], family)
        assert (X[0] != X[1]).nnz == 0
    X = matrix([words[0], words[2]], vocabs['all_position'], 'all_position')
    assert (X[0] != X[1]).nnz > 0
    X4 = matrix([words[0], words[4]], vocabs['terminal_subset_1to4'], 'terminal_subset_1to4')
    X5 = matrix([words[0], words[4]], vocabs['terminal_tagged_1to5'], 'terminal_tagged_1to5')
    assert (X4[0] != X4[1]).nnz == 0 and (X5[0] != X5[1]).nnz > 0
    assert features('ab', 'terminal_subset_1to4') == {'b$', 'ab$', '^ab$'}
    assert features('ab', 'terminal_tagged_1to5') == {'S1:b', 'S2:ab'}
    v = vocabulary(['abc', 'xbc'], 'terminal_tagged_1to5')
    assert 'S3:abc' not in v and 'S2:bc' in v
    before = dict(v)
    matrix(['qqqqq'], v, 'terminal_tagged_1to5')
    assert v == before
    assert longest_terminal('abc', v, 'terminal_tagged_1to5') == 2
    assert longest_terminal('qqq', v, 'terminal_tagged_1to5') == 0
    assert sn.selective('A\u0301ÇÃÊÀÜ') == 'açãeàü'
    assert sp.stripped('A\u0301ÇÃÊÀÜ') == 'acaeau'
    assert sp.assignment('sintético') == sp.assignment('SINTETICO')
    y = np.array([0, 1, 2])
    a = np.eye(3)[[0, 1, 0]]
    b = np.eye(3)[[1, 1, 2]]
    pair = paired_counts(y, a, b)
    assert pair['both_correct'] == pair['only_a_correct'] == pair['only_b_correct'] == 1
    assert pair['both_wrong'] == 0 and pair['difference'] == 0
    m = metrics(y, np.eye(3)[[0, 0, 2]])
    np.testing.assert_allclose(m['macro_f1'], (2 / 3 + 0 + 1) / 3)
    np.testing.assert_allclose(m['macro_recall'], 2 / 3)
    assert metrics(np.array([0]), np.array([[1., 0., 0.]]))['macro_recall'] == 1 / 3
    assert ending_stratum('xxico') == 'ends_ico_ica_icos_icas'
    assert ending_stratum('xxlogo') == 'ends_logo_loga_logos_logas'
    assert ending_stratum('xxosos') == 'ends_oso_osa_osos_osas'
    assert ending_stratum('xxabc').startswith('remainder')
    return {'status': 'PASS', 'fixtures': 'explicitly synthetic strings/probability tables',
            'checks': ['exact subset vocabulary', 'own-family L2 norms and zero vectors',
                       'terminal vector invariance to nonterminal changes',
                       'five-letter sensitivity beyond four-letter control',
                       'short words and start/end boundaries', 'train-only min-frequency threshold',
                       'query cannot mutate vocabulary', 'Unicode and split grouping',
                       'paired arithmetic', 'explicit macro-F1/macro-recall definitions',
                       'predeclared ending groups'], 'original_utility_checks': sp.checks()}

