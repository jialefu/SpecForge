import json
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from scripts.prepare_data import process_and_save_ds


def _process_row(row, dataset_name=None):
    return (
        {
            "id": str(row["id"]),
            "conversations": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
        },
        0,
    )


def _toy_dataset():
    return Dataset.from_list(
        [
            {"id": 1, "prompt": "hello", "response": "hi"},
            {"id": 2, "prompt": "bye", "response": "see you"},
        ]
    )


class TestPrepareDataOutputFormats(unittest.TestCase):
    def test_json_output_is_default_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            process_and_save_ds(
                _toy_dataset(), None, output_path, _process_row, "toy"
            )

            jsonl_path = output_path / "toy_train.jsonl"
            self.assertTrue(jsonl_path.exists())
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["conversations"][0]["role"], "user")
            self.assertEqual(rows[0]["conversations"][1]["content"], "hi")

    def test_parquet_output_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            process_and_save_ds(
                _toy_dataset(),
                None,
                output_path,
                _process_row,
                "toy",
                output_format="parquet",
            )

            parquet_path = output_path / "toy_train.parquet"
            self.assertTrue(parquet_path.exists())
            loaded = load_dataset(
                "parquet", data_files=str(parquet_path), split="train"
            )
            self.assertEqual(loaded[1]["conversations"][0]["content"], "bye")

    def test_hf_dataset_output_can_be_loaded_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            process_and_save_ds(
                _toy_dataset(),
                None,
                output_path,
                _process_row,
                "toy",
                output_format="hf-dataset",
            )

            dataset_path = output_path / "toy_train"
            self.assertTrue(dataset_path.is_dir())
            loaded = load_from_disk(str(dataset_path))
            self.assertEqual(loaded[0]["conversations"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
