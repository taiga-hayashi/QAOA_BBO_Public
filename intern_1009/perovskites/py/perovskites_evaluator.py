"""Pinned Olympus Perovskites lookup and strict problem-specific codec.

No Olympus, FM, solver, or QAOA dependency. Vector position q is qubit q
(LSB first when converted to a computational-basis integer).
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from numbers import Real
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PerovskitesEvaluator:
    def __init__(self, data_path=None, config_path=None):
        self.data_path = Path(data_path) if data_path is not None else ROOT / 'data/reference/data.csv'
        self.config_path = Path(config_path) if config_path is not None else ROOT / 'data/reference/config.json'
        config = json.loads(self.config_path.read_text())
        if [p['name'] for p in config['parameters']] != ['organic', 'cation', 'anion']:
            raise ValueError('Unexpected parameter schema')
        if config['default_goal'] != 'minimize' or [p['name'] for p in config['measurements']] != ['hse_gap']:
            raise ValueError('Unexpected objective schema')
        self.options = tuple(tuple(p['options']) for p in config['parameters'])
        if tuple(map(len, self.options)) != (16, 3, 4) or any(len(set(o)) != len(o) for o in self.options):
            raise ValueError('Unexpected or duplicate category definitions')
        self.group_sizes = tuple(map(len, self.options))
        self.n_onehot_variables = sum(self.group_sizes)
        self._values = {}
        self.row_records = []
        with self.data_path.open(newline='') as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if len(row) != 4 or any(not cell.strip() or cell != cell.strip() for cell in row):
                    raise ValueError(f'Row {line}: expected four nonempty, unpadded columns')
                key = tuple(row[:3])
                if any(label not in options for label, options in zip(key, self.options)):
                    raise ValueError(f'Row {line}: unknown category')
                if key in self._values:
                    raise ValueError(f'Row {line}: duplicate categorical key')
                try:
                    value = float(row[3])
                except ValueError as exc:
                    raise ValueError(f'Row {line}: invalid objective') from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'Row {line}: nonfinite or negative band gap')
                self._values[key] = value
                self.row_records.append({'row': line, 'categories': list(key), 'raw_value': row[3], 'hse_gap': value})
        expected = set(itertools.product(*self.options))
        if set(self._values) != expected or len(self.row_records) != 192:
            raise ValueError('Dataset does not cover the complete 192-candidate Cartesian product')
        self.data_sha256 = hashlib.sha256(self.data_path.read_bytes()).hexdigest()
        self.config_sha256 = hashlib.sha256(self.config_path.read_bytes()).hexdigest()

    def _key(self, categories):
        if isinstance(categories, (str, bytes)):
            raise ValueError('Expected three categorical labels')
        try:
            key = tuple(categories)
        except TypeError as exc:
            raise ValueError('Expected three categorical labels') from exc
        if len(key) != 3 or any(not isinstance(k, str) or k not in o for k, o in zip(key, self.options)):
            raise ValueError('Expected valid organic, cation, anion labels in config order')
        return key

    def evaluate(self, categories):
        return self._values[self._key(categories)]

    def encode(self, categories):
        key = self._key(categories)
        bits = []
        for label, options in zip(key, self.options):
            bits.extend(int(i == options.index(label)) for i in range(len(options)))
        return tuple(bits)

    def decode(self, bits):
        if isinstance(bits, (str, bytes)):
            raise ValueError('Expected a numeric 23-bit vector, q0 first')
        try:
            vector = tuple(bits)
        except TypeError as exc:
            raise ValueError('Expected a numeric 23-bit vector') from exc
        if len(vector) != self.n_onehot_variables or any(
            not isinstance(b, Real) or not math.isfinite(b) or b not in (0, 1) for b in vector
        ):
            raise ValueError('Expected exactly 23 finite binary entries')
        labels = []
        offset = 0
        for options in self.options:
            block = vector[offset:offset + len(options)]
            if sum(block) != 1:
                raise ValueError('Each One-Hot group must contain exactly one active entry')
            labels.append(options[block.index(1)])
            offset += len(options)
        return tuple(labels)

    def evaluate_onehot(self, bits):
        return self.evaluate(self.decode(bits))

    def candidates(self):
        """Enumerate labels only; objective values remain behind evaluate()."""
        return itertools.product(*self.options)

    def basis_index(self, bits):
        if isinstance(bits, (str, bytes)):
            raise ValueError('Expected a numeric 23-bit vector')
        try:
            vector = tuple(bits)
        except TypeError as exc:
            raise ValueError('Expected a numeric 23-bit vector') from exc
        self.decode(vector)
        return sum(int(b) << q for q, b in enumerate(vector))
