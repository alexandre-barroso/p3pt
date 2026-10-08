"""Original exploratory stress-prediction experiment; aggregate exports only.

Reads public PSL in place. Does not import protected code or export any rows,
model parameters, vocabulary, or source assets. See the prospective protocol.
"""


from __future__ import annotations


import argparse


from collections import Counter, defaultdict


from datetime import datetime, timezone


import csv


import hashlib


import json


from pathlib import Path


import sys


import unicodedata as ud


import numpy as np


import scipy


from scipy import sparse


from scipy.optimize import minimize


from scipy.special import logsumexp


ALPHABET = set('abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç')


PSL_SHA = '96e083977e093a9e3550d2feee2ec725ac0768d4bff386a295480bafdeabe0e6'


NAMESPACE = 'project8-stress-followup-v1\0'


def canonical(w):
    return ud.normalize('NFC', w.casefold())


def stripped(w):
    return ud.normalize('NFC', ''.join(c for c in ud.normalize('NFD', canonical(w)) if not ud.combining(c)))


def assignment(w):
    rank = int.from_bytes(hashlib.sha256((NAMESPACE + stripped(w)).encode()).digest(), 'big')
    # Integer arithmetic avoids floating-point boundary ambiguity.
    return 0 if 20 * rank < 14 * 2**256 else 1 if 20 * rank < 17 * 2**256 else 2


def read_types(path):
    byword = defaultdict(set)
    exclusions = Counter()
    rows = 0
    with path.open() as f:
        for r in csv.DictReader(f):
            rows += 1
            w = canonical(r['word'])
            syll = r['pro'].split('-')
            marked = [i for i, s in enumerate(syll) if s.startswith("'")]
            label = {'final': 0, 'penult': 1, 'antepenult': 2}.get(r['stressLoc'])
            if not w or not set(w) <= ALPHABET:
                exclusions['nonstandard_or_compound_orthography'] += 1
            elif label is None or not all(syll) or len(marked) != 1 or r['pro'].count("'") != 1 or len(syll) - marked[0] != label + 1 or len(syll) != int(r['nSyl']):
                exclusions['disagreeing_annotation_fields'] += 1
            else:
                byword[w].add(label)
    selected = sorted((w, next(iter(ys))) for w, ys in byword.items() if len(ys) == 1)
    exclusions['conflicting_stress_types'] = sum(len(ys) > 1 for ys in byword.values())
    return selected, {'input_rows': rows, 'screened_unique_types': len(byword), 'selected_unique_types': len(selected), 'exclusions': dict(exclusions)}


def grams(w):
    marked = '^' + w + '$'
    return {marked[i:i+n] for n in range(2, 6) for i in range(len(marked)-n+1)}


def vectorizer(train_words):
    freq = Counter(g for w in train_words for g in grams(w))
    return {g: i for i, g in enumerate(sorted(g for g, n in freq.items() if n >= 2))}


def matrix(words, vocabulary):
    indptr, indices, data = [0], [], []
    for w in words:
        ix = sorted(vocabulary[g] for g in grams(w) if g in vocabulary)
        indices.extend(ix)
        data.extend([1 / np.sqrt(len(ix))] * len(ix) if ix else [])
        indptr.append(len(indices))
    return sparse.csr_matrix((np.array(data), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)), shape=(len(words), len(vocabulary)))


def objective(theta, X, y, C):
    width = X.shape[1]
    W = theta[:width * 3].reshape(width, 3)
    bias = theta[width * 3:]
    scores = X @ W + bias
    logp = scores - logsumexp(scores, axis=1, keepdims=True)
    value = -logp[np.arange(len(y)), y].mean() + np.sum(W * W) / (2 * C * len(y))
    delta = np.exp(logp)
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    grad = np.concatenate(((X.T @ delta + W / (C * len(y))).ravel(), delta.sum(axis=0)))
    return value, grad


def fit(X, y, C):
    theta = np.zeros(X.shape[1] * 3 + 3)
    counts = np.bincount(y, minlength=3)
    theta[-3:] = np.log(counts / len(y))
    fit = minimize(objective, theta, args=(X, y, C), method='L-BFGS-B', jac=True,
                   options={'maxiter': 1000, 'gtol': 1e-6, 'ftol': 1e-11, 'maxcor': 10})
    maximum = float(np.max(np.abs(fit.jac)))
    record = {'C': C, 'success': bool(fit.success), 'message': str(fit.message), 'iterations': int(fit.nit), 'objective': float(fit.fun), 'max_abs_gradient': maximum}
    if not fit.success or maximum > 1e-5:
        raise RuntimeError('Optimizer gate failed: ' + json.dumps(record))
    return fit.x, record


def probabilities(theta, X):
    scores = X @ theta[:-3].reshape(X.shape[1], 3) + theta[-3:]
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def metrics(y, probabilities):
    pred = probabilities.argmax(axis=1)
    conf = np.zeros((3, 3), dtype=int)
    np.add.at(conf, (y, pred), 1)
    tp = np.diag(conf)
    true = conf.sum(axis=1)
    estimated = conf.sum(axis=0)
    f1 = np.divide(2 * tp, true + estimated, out=np.zeros(3, dtype=float), where=true + estimated > 0)
    recall = np.divide(tp, true, out=np.zeros(3, dtype=float), where=true > 0)
    return {'observations': len(y), 'accuracy': float(np.mean(y == pred)), 'macro_f1': float(f1.mean()), 'log_loss': float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-300)).mean()), 'confusion_true_rows': conf.tolist(), 'support_by_class': true.tolist(), 'predictions_by_class': estimated.tolist(), 'recall_by_class': recall.tolist(), 'f1_by_class': f1.tolist()}


def oracle(words, labels):
    counts = defaultdict(Counter)
    for w, y in zip(words, labels):
        counts[w][int(y)] += 1
    errors = sum(sum(c.values()) - max(c.values()) for c in counts.values())
    return {'minimum_errors': errors, 'minimum_error_rate': errors / len(words), 'ambiguous_keys': sum(len(c) > 1 for c in counts.values()), 'scope': 'Finite test-label lookup oracle, not a trained model or population bound'}


def suffix_model(train_words, y, queries, size):
    global_count = np.bincount(y, minlength=3).astype(float) + 1
    counts = defaultdict(lambda: np.zeros(3))
    for w, label in zip(train_words, y):
        counts[w[-size:]][label] += 1
    return np.array([(counts[w[-size:]] + 1) / (counts[w[-size:]].sum() + 3) if w[-size:] in counts else global_count / global_count.sum() for w in queries])


def group_interval(words, y, pa, pb, seed=20260926):
    groups = defaultdict(list)
    for i, w in enumerate(words):
        groups[stripped(w)].append(i)
    units = list(groups.values())
    sizes = np.array([len(g) for g in units])
    changes = (pa.argmax(axis=1) == y).astype(int) - (pb.argmax(axis=1) == y).astype(int)
    sums = np.array([changes[g].sum() for g in units])
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(2000):
        draws = rng.integers(len(units), size=len(units))
        resampled.append(float(sums[draws].sum() / sizes[draws].sum()))
    return {'accuracy_difference': float(changes.mean()), 'conditional_group_bootstrap_95_percentile': np.quantile(resampled, [.025, .975]).tolist(), 'groups': len(units), 'draws': 2000, 'seed': seed, 'scope': 'Conditional on frozen split and fits; not training, tuning or participant uncertainty'}


def checks():
    assert stripped('A\u0301ção') == 'acao'
    assert assignment('sintético') == assignment('SINTETICO')
    vocab = vectorizer(['ab', 'ab', 'cd'])
    assert 'cd' not in vocab
    X = matrix(['ab', 'zz', 'ab'], vocab)
    assert X[1].nnz == 0
    np.testing.assert_allclose(np.array(X.multiply(X).sum(axis=1)).ravel(), [1, 0, 1])
    y = np.array([0, 1, 2])
    theta = np.linspace(-.2, .3, X.shape[1] * 3 + 3)
    _, actual = objective(theta, X, y, 1)
    numeric = []
    for j in range(len(theta)):
        step = np.zeros_like(theta); step[j] = 1e-6
        numeric.append((objective(theta + step, X, y, 1)[0] - objective(theta - step, X, y, 1)[0]) / 2e-6)
    np.testing.assert_allclose(actual, numeric, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(probabilities(theta, X).sum(axis=1), 1)
    assert oracle(['a', 'a', 'b'], [0, 1, 2])['minimum_errors'] == 1
    return {'finite_difference_gradient_coordinates': len(theta), 'status': 'PASS'}

