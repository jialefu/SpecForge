import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

_LOADING_PATH = Path(__file__).parents[2] / "specforge" / "data" / "loading.py"
_SPEC = importlib.util.spec_from_file_location("specforge_data_loading", _LOADING_PATH)
_LOADING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOADING)
load_conversation_dataset = _LOADING.load_conversation_dataset


def _rows():
    return [
        {
            "id": "a",
            "status": "success",
            "conversations": [
                {"role": "user", "content": "hello", "thinking": None},
                {
                    "role": "assistant",
                    "content": "hi",
                    "thinking": None,
                    "extra": "drop me",
                },
            ],
        },
        {
            "id": "b",
            "status": "success",
            "conversations": [
                {"role": "user", "content": "question", "thinking": None},
                {"role": "assistant", "content": "answer", "thinking": None},
            ],
        },
    ]


class TestConversationDatasetLoading(unittest.TestCase):
    def assert_normalized(self, dataset):
        self.assertEqual(len(dataset), 2)
        row = dataset[0]
        self.assertEqual(set(row.keys()), {"id", "conversations"})
        self.assertEqual(row["id"], "a")
        self.assertEqual(
            row["conversations"],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        )

    def test_loads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in _rows():
                    f.write(json.dumps(row) + "\n")

            dataset = load_conversation_dataset(str(path))

        self.assert_normalized(dataset)

    def test_loads_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.parquet"
            Dataset.from_list(_rows()).to_parquet(str(path))

            dataset = load_conversation_dataset(str(path))

        self.assert_normalized(dataset)

    def test_loads_saved_hf_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "saved"
            Dataset.from_list(_rows()).save_to_disk(str(path))

            dataset = load_conversation_dataset(str(path))

        self.assert_normalized(dataset)

    def test_missing_local_path_explains_hf_fallback_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = str(Path(tmpdir) / "missing")
            with patch.object(
                _LOADING, "load_dataset", side_effect=RuntimeError("boom")
            ):
                with self.assertWarnsRegex(
                    UserWarning, "Trying to load it as a Hugging Face dataset id"
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "loading it as a Hugging Face dataset id also failed",
                    ):
                        load_conversation_dataset(missing_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
