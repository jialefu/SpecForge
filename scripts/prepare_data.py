import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from specforge.data.dataset_recipes import DATASET_RECIPE_REGISTRY, load_recipe

"""
This script will convert the ultrachat/sharegpt dataset to the following schema in jsonl format:
{
    "id": str,
    "conversations": [
        {
            "role": str,
            "content": str
        }
    ],
}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_RECIPE_REGISTRY.get_all_dataset_names(),
        help="The demo dataset to quickly run the training for speculative decoding",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="The path to save the processed dataset, if not specified, the dataset will be saved in the cache/dataset/dataset_name directory of the root path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="The number of samples to process from the dataset, if not specified, all samples will be processed",
    )
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Whether to split the dataset into train and eval sets, default is False",
    )
    return parser.parse_args()


def process_and_save_ds(train_ds, test_ds, output_path, proc_fn, dataset_name):
    train_output_jsonl_path = output_path.joinpath(f"{dataset_name}_train.jsonl")
    if train_output_jsonl_path.exists():
        print(
            f"The dataset {dataset_name} has already been processed and saved in {train_output_jsonl_path}, skipping..."
        )
        return

    total_skipped_count = 0
    with open(train_output_jsonl_path, "w") as f:
        for item in tqdm(train_ds, desc=f"Processing {dataset_name} dataset"):
            row, skipped_count = proc_fn(item, dataset_name)
            if row is None:
                continue
            total_skipped_count += skipped_count
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if test_ds is not None:
        test_output_jsonl_path = output_path.joinpath(f"{dataset_name}_test.jsonl")
        with open(test_output_jsonl_path, "w") as f:
            for item in tqdm(test_ds, desc=f"Processing {dataset_name} test dataset"):
                row, skipped_count = proc_fn(item, dataset_name)
                if row is None:
                    continue
                total_skipped_count += skipped_count
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if total_skipped_count > 0:
        total_messages = len(train_ds) + (len(test_ds) if test_ds is not None else 0)
        print(
            f"Skipped {total_skipped_count}/{total_messages} messages for {dataset_name}"
        )


def main():
    args = parse_args()
    recipe = DATASET_RECIPE_REGISTRY.get(args.dataset)
    ds = load_recipe(recipe, args.dataset)
    proc_fn = recipe.processor

    if args.sample_size is not None and args.sample_size < len(ds):
        ds = ds.select(range(args.sample_size))
        print(f"Processing {args.sample_size} samples from the dataset {args.dataset}")
    if args.split_eval:
        ds = ds.train_test_split(test_size=0.05)
        train_ds = ds["train"]
        test_ds = ds["test"]
    else:
        train_ds = ds
        test_ds = None

    if args.output_path is None:
        output_path = ROOT_PATH.joinpath("cache", "dataset")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    process_and_save_ds(train_ds, test_ds, output_path, proc_fn, args.dataset)


if __name__ == "__main__":
    main()
