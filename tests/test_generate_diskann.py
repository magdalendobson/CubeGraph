import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_diskann as generator
import generate_metadata as original


class DiskANNGeneratorTests(unittest.TestCase):
    def arguments(self, output, **overrides):
        values = dict(num_base=40, num_queries=7, dimensions=2,
                      distribution="uniform", box_volume=0.01, seed=42,
                      output_dir=output)
        values.update(overrides)
        return argparse.Namespace(**values)

    def read_jsonl(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_all_distributions_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for distribution in generator.DISTRIBUTIONS:
                for dimensions in (2, 3, 4):
                    with self.subTest(distribution=distribution, dimensions=dimensions):
                        output = Path(tmp) / f"{distribution}-{dimensions}"
                        args = self.arguments(output, distribution=distribution,
                                              dimensions=dimensions)
                        generator.generate(args)
                        labels = self.read_jsonl(output / "data_labels.jsonl")
                        queries = self.read_jsonl(output / "query_predicates.jsonl")
                        manifest = json.loads((output / "manifest.json").read_text())
                        fields = [f"a{i}" for i in range(dimensions)]
                        self.assertEqual(len(labels), args.num_base)
                        self.assertEqual(len(queries), args.num_queries)
                        self.assertEqual(manifest["fields"], fields)
                        self.assertEqual(manifest["num_base"], args.num_base)
                        self.assertEqual(manifest["num_queries"], args.num_queries)
                        data = np.array([[row[f] for f in fields] for row in labels])
                        self.assertTrue(np.isfinite(data).all())
                        self.assertTrue(((0 <= data) & (data <= 1)).all())
                        np.testing.assert_array_equal(data, data.astype(np.float32))
                        for doc_id, row in enumerate(labels):
                            self.assertEqual(row["doc_id"], doc_id)
                            self.assertEqual(set(row), {"doc_id", *fields})
                        domain = data.max(axis=0) - data.min(axis=0)
                        for query_id, row in enumerate(queries):
                            self.assertEqual(set(row), {"query_id", "filter"})
                            self.assertEqual(row["query_id"], query_id)
                            terms = row["filter"]["$and"]
                            self.assertEqual(len(terms), dimensions * 2)
                            lo = np.array([terms[2*i][f]["$gte"]
                                           for i, f in enumerate(fields)])
                            hi = np.array([terms[2*i+1][f]["$lte"]
                                           for i, f in enumerate(fields)])
                            self.assertTrue((lo >= data.min(axis=0)).all())
                            self.assertTrue((hi <= data.max(axis=0)).all())
                            self.assertTrue((lo <= hi).all())
                            self.assertAlmostEqual(np.prod((hi - lo) / domain),
                                                   args.box_volume, places=7)

    def test_reuses_original_distributions(self):
        for distribution in generator.DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                actual = generator.make_metadata(23, 3, distribution, 19)
                np.random.seed(19)
                function = getattr(original, f"generate_{distribution}")
                if distribution in ("uniform", "normal"):
                    expected = function(23, 3)
                else:
                    expected = function(23, 3, seed=19)
                np.testing.assert_array_equal(actual, expected.astype(np.float32))

    def test_reproducibility_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for distribution in generator.DISTRIBUTIONS:
                outputs = [Path(tmp) / f"{distribution}-{i}" for i in range(3)]
                for i, output in enumerate(outputs):
                    generator.generate(self.arguments(output, distribution=distribution,
                                                      seed=42 if i < 2 else 43))
                for name in ("data_labels.jsonl", "query_predicates.jsonl", "manifest.json"):
                    self.assertEqual((outputs[0] / name).read_bytes(),
                                     (outputs[1] / name).read_bytes())
                    self.assertNotEqual((outputs[0] / name).read_bytes(),
                                        (outputs[2] / name).read_bytes())

    def test_full_domain_and_inclusive_boundaries(self):
        data = np.array([[0, 0.2], [0.4, 0.8], [1, 1]], dtype=np.float32)
        for lo, hi in generator.make_boxes(data, 3, 1.0, 42):
            np.testing.assert_array_equal(lo, data.min(axis=0))
            np.testing.assert_array_equal(hi, data.max(axis=0))
            predicate = generator.box_predicate(["a0", "a1"], lo, hi)["$and"]
            for row in data:
                self.assertTrue(all(
                    row[i] >= predicate[2*i][f"a{i}"]["$gte"]
                    and row[i] <= predicate[2*i+1][f"a{i}"]["$lte"]
                    for i in range(2)
                ))

    def test_small_base_counts(self):
        for distribution in generator.DISTRIBUTIONS:
            for count in (1, 3):
                with self.subTest(distribution=distribution, count=count):
                    data = generator.make_metadata(count, 4, distribution, 0)
                    self.assertEqual(data.shape, (count, 4))
                    for lo, hi in generator.make_boxes(data, 2, 0.01, 0):
                        self.assertTrue(np.isfinite(lo).all())
                        self.assertTrue(np.isfinite(hi).all())
                        self.assertTrue((lo <= hi).all())

    def test_uniform_realized_fraction(self):
        data = generator.make_metadata(20000, 2, "uniform", 42)
        total = sum(np.all((data >= lo) & (data <= hi), axis=1).sum()
                    for lo, hi in generator.make_boxes(data, 100, 0.01, 42))
        self.assertAlmostEqual(total / (len(data) * 100), 0.01, delta=0.001)

    def test_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output / "data_labels.jsonl"
            sentinel.write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                generator.generate(self.arguments(output))
            self.assertEqual(sentinel.read_text(), "do not overwrite")
            self.assertFalse((output / "query_predicates.jsonl").exists())

    def test_cli_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "generate_diskann.py"),
                 "--num-base", "12", "--num-queries", "4"],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(tmp) / "diskann-data"
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["dimensions"], 2)
            self.assertEqual(manifest["distribution"], "uniform")
            self.assertEqual(manifest["box_volume"], 0.01)
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(len(self.read_jsonl(output / "data_labels.jsonl")), 12)
            self.assertEqual(len(self.read_jsonl(output / "query_predicates.jsonl")), 4)

    def test_cli_invalid_inputs(self):
        cases = [
            ("--num-base", "0"), ("--num-base", "-1"),
            ("--num-queries", "0"), ("--num-queries", "-2"),
            ("--dimensions", "5"), ("--distribution", "unknown"),
            ("--box-volume", "0"), ("--box-volume", "-0.1"),
            ("--box-volume", "1.1"), ("--box-volume", "nan"),
            ("--box-volume", "inf"), ("--seed", "-1"),
            ("--seed", "4294967296"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "invalid"
            for flag, value in cases:
                with self.subTest(flag=flag, value=value):
                    result = subprocess.run(
                        [sys.executable, str(SCRIPTS / "generate_diskann.py"),
                         "--num-base", "12", "--num-queries", "4",
                         "--output-dir", str(output), flag, value],
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("error:", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
