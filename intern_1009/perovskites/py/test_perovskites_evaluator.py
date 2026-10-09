"""Meaningful regression checks for data corruption and input rejection."""
import csv
import tempfile
import unittest
from pathlib import Path
from perovskites_evaluator import PerovskitesEvaluator, ROOT


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = PerovskitesEvaluator()

    def test_all_rows_against_independent_csv_reader(self):
        with (ROOT / 'data/reference/data.csv').open(newline='') as stream:
            for row in csv.reader(stream):
                labels = tuple(row[:3])
                bits = self.evaluator.encode(labels)
                self.assertEqual(self.evaluator.decode(bits), labels)
                self.assertEqual(self.evaluator.evaluate(labels), float(row[3]))
                self.assertEqual(self.evaluator.evaluate_onehot(bits), float(row[3]))
        self.assertEqual(len(list(self.evaluator.candidates())), 192)

    def test_known_first_row_and_bit_positions(self):
        bits = self.evaluator.encode(('ethylammonium', 'Ge', 'F'))
        self.assertEqual([q for q, b in enumerate(bits) if b], [0, 16, 19])
        self.assertEqual(self.evaluator.basis_index(bits), 1 + (1 << 16) + (1 << 19))
        self.assertEqual(self.evaluator.basis_index(iter(bits)), self.evaluator.basis_index(bits))
        self.assertEqual(self.evaluator.evaluate_onehot(bits), 5.3704)

    def test_last_category_positions(self):
        labels = ('imidazolium', 'Pb', 'I')
        bits = self.evaluator.encode(labels)
        self.assertEqual([q for q, b in enumerate(bits) if b], [15, 18, 22])
        self.assertEqual(self.evaluator.decode(bits), labels)

    def test_invalid_categories(self):
        for value in [('unknown', 'Ge', 'F'), ('ethylammonium', 'F', 'Ge'), ('Ge',), 'Ge', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluator.evaluate(value)

    def test_invalid_onehot(self):
        good = list(self.evaluator.encode(('ethylammonium', 'Ge', 'F')))
        variants = [good[:-1], good + [0], [0]*23, '0'*23, None]
        for replacement in [2, 0.5, float('nan'), float('inf'), '1', [1]]:
            bad = good.copy(); bad[0] = replacement; variants.append(bad)
        bad = good.copy(); bad[1] = 1; variants.append(bad)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluator.evaluate_onehot(value)

    def test_corrupted_data_rejected(self):
        rows = (ROOT / 'data/reference/data.csv').read_text().splitlines()
        mutations = {'missing': rows[:-1], 'duplicate': rows + [rows[0]],
                     'unknown': ['unknown,Ge,F,5.3704'] + rows[1:],
                     'nan': ['ethylammonium,Ge,F,nan'] + rows[1:],
                     'negative': ['ethylammonium,Ge,F,-1'] + rows[1:],
                     'empty': ['ethylammonium,Ge,F,'] + rows[1:],
                     'header': ['organic,cation,anion,hse_gap'] + rows}
        with tempfile.TemporaryDirectory() as tmp:
            for name, mutated in mutations.items():
                path = Path(tmp) / f'{name}.csv';path.write_text('\n'.join(mutated)+'\n')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    PerovskitesEvaluator(data_path=path)


if __name__ == '__main__':
    unittest.main()
