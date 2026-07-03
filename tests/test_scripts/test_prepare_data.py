import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.utils import execute_shell_command

from specforge.data import dataset_processors, dataset_recipes
from scripts import prepare_data

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


class TestPrepareData(unittest.TestCase):
    def test_load_recipe_single_source(self):
        recipe = dataset_recipes.DatasetRecipe(
            sources=[
                {
                    "repo": "mock/repo",
                    "subset": "mock_subset",
                    "split": "train",
                    "trust_remote_code": True,
                }
            ],
            processor=dataset_processors.identity_processor,
        )
        ds = object()

        with patch(
            "specforge.data.dataset_recipes.load_dataset", return_value=ds
        ) as mock_load:
            self.assertIs(dataset_recipes.load_recipe(recipe), ds)

        mock_load.assert_called_once_with(
            "mock/repo",
            "mock_subset",
            split="train",
            trust_remote_code=True,
        )

    def test_load_recipe_concatenates_multiple_sources(self):
        recipe = dataset_recipes.DatasetRecipe(
            sources=[
                {"repo": "mock/repo", "subset": "a", "split": "train"},
                {"repo": "mock/repo", "subset": "b", "split": "train"},
            ],
            processor=dataset_processors.identity_processor,
        )
        first_ds = object()
        second_ds = object()
        merged_ds = object()

        with (
            patch(
                "specforge.data.dataset_recipes.load_dataset",
                side_effect=[first_ds, second_ds],
            ),
            patch(
                "specforge.data.dataset_recipes.concatenate_datasets",
                return_value=merged_ds,
            ) as mock_concatenate,
        ):
            self.assertIs(dataset_recipes.load_recipe(recipe), merged_ds)

        mock_concatenate.assert_called_once_with([first_ds, second_ds])

    def test_opc_all_expands_all_opc_subsets(self):
        recipe = dataset_recipes.DATASET_RECIPE_REGISTRY.get("opc_all")

        self.assertEqual(
            [source["subset"] for source in recipe.sources],
            [
                "largescale_diverse_instruct",
                "filtered_infinity_instruct",
                "realuser_instruct",
            ],
        )
        self.assertTrue(
            all(
                source["repo"] == "OpenCoder-LLM/opc-sft-stage1"
                for source in recipe.sources
            )
        )

    def test_identity_processor_writes_normalized_rows(self):
        rows = [
            {
                "id": "row-1",
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            prepare_data.process_and_save_ds(
                rows,
                None,
                output_path,
                dataset_processors.identity_processor,
                "normalized",
            )

            output_file = output_path.joinpath("normalized_train.jsonl")
            self.assertTrue(output_file.exists())
            self.assertEqual(json.loads(output_file.read_text().strip()), rows[0])

    def test_prepare_sharegpt(self):
        sharegpt_train_path = CACHE_DIR.joinpath("dataset", "sharegpt_train.jsonl")

        if sharegpt_train_path.exists():
            # delete the file
            sharegpt_train_path.unlink()
        process = execute_shell_command(
            f"{sys.executable} scripts/prepare_data.py --dataset sharegpt"
        )
        process.wait()
        self.assertEqual(process.returncode, 0)
        self.assertTrue(sharegpt_train_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
