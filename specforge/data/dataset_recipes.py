from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from datasets import concatenate_datasets, load_dataset

from specforge.data.dataset_processors import (
    add_index,
    ensure_allava4v_assets,
    identity_processor,
    process_camel_row,
    process_codealpaca_row,
    process_gsm8k_row,
    process_hendrycks_math_row,
    process_magicoder_evol_instruct_row,
    process_math_qa_row,
    process_nebius_infinity_instruct,
    process_opc_sft_stage1,
    process_opencodeinstruct_row,
    process_sciq_row,
    process_sharegpt4v_row,
    process_sharegpt_row,
    process_ultrachat_row,
    raise_unsupported_sharegpt4v,
)


Processor = Callable[[Dict, str | None], Tuple[Dict | None, int]]
PostLoad = Callable[[Any], Any]


@dataclass(frozen=True)
class DatasetRecipe:
    sources: list[dict]
    processor: Processor
    post_load: PostLoad | None = None


class DatasetRecipeRegistry:
    def __init__(self):
        self.recipes = {}

    def register(self, name: str, recipe: DatasetRecipe, override: bool = False):
        assert (
            override or name not in self.recipes
        ), f"Dataset recipe {name} has already been registered"
        self.recipes[name] = recipe

    def get(self, name: str) -> DatasetRecipe:
        return self.recipes[name]

    def get_all_dataset_names(self) -> List[str]:
        return list(self.recipes.keys())


def load_source(source: dict):
    source = source.copy()
    repo = source.pop("repo")
    subset = source.pop("subset", None)
    split = source.pop("split", None)
    if subset is None:
        return load_dataset(repo, split=split, **source)
    return load_dataset(repo, subset, split=split, **source)


def load_recipe(recipe: DatasetRecipe, dataset_name: str | None = None):
    if not recipe.sources:
        name = dataset_name or "<unnamed>"
        raise ValueError(f"Dataset recipe '{name}' has no sources.")

    datasets = [load_source(source) for source in recipe.sources]
    ds = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if recipe.post_load is not None:
        ds = recipe.post_load(ds)
    return ds


OPC_REPO = "OpenCoder-LLM/opc-sft-stage1"
OPC_SUBSETS = [
    "largescale_diverse_instruct",
    "filtered_infinity_instruct",
    "realuser_instruct",
]

HENDRYCKS_MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


DATASET_RECIPE_REGISTRY = DatasetRecipeRegistry()

DATASET_RECIPE_REGISTRY.register(
    name="ultrachat",
    recipe=DatasetRecipe(
        sources=[{"repo": "HuggingFaceH4/ultrachat_200k", "split": "train_sft"}],
        processor=process_ultrachat_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="sharegpt",
    recipe=DatasetRecipe(
        sources=[{"repo": "Aeala/ShareGPT_Vicuna_unfiltered", "split": "train"}],
        processor=process_sharegpt_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="eaglechat",
    recipe=DatasetRecipe(
        sources=[{"repo": "zhaode/EagleChat", "split": "train"}],
        processor=identity_processor,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend",
    recipe=DatasetRecipe(
        sources=[{"repo": "mlabonne/open-perfectblend", "split": "train"}],
        processor=process_sharegpt_row,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend-llama3.1-8b-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct",
                "split": "train",
            }
        ],
        processor=identity_processor,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend-llama3.3-70b-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "frankleeeee/PerfectBlend-Regenerated-Llama-3.3-70B-Instruct",
                "split": "train",
            }
        ],
        processor=identity_processor,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend-llama4-scout-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "frankleeeee/PerfectBlend-Regenerated-Llama-4-Scout-17B-16E-Instruct",
                "split": "train",
            }
        ],
        processor=identity_processor,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend-llama4-maverick-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "frankleeeee/PerfectBlend-Regenerated-Llama-4-Maverick-17B-128E-Instruct",
                "split": "train",
            }
        ],
        processor=identity_processor,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="perfectblend-qwen3-8b",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "parquet",
                "data_files": {
                    "train": "hf://datasets/jihwan1205/perfectblend-qwen3-8b-regen/data/*.parquet"
                },
                "split": "train",
            }
        ],
        processor=identity_processor,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="magpie-qwen2.5-pro-1m-v0.1",
    recipe=DatasetRecipe(
        sources=[
            {"repo": "Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1", "split": "train"}
        ],
        processor=process_sharegpt_row,
        post_load=lambda ds: ds.rename_column("uuid", "id"),
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="sharegpt4v",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "Lin-Chen/ShareGPT4V",
                "subset": "ShareGPT4V",
                "split": "train",
            }
        ],
        processor=process_sharegpt4v_row,
        post_load=raise_unsupported_sharegpt4v,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="allava4v",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "FreedomIntelligence/ALLaVA-4V",
                "subset": "allava_laion",
                "split": "instruct",
            }
        ],
        processor=process_sharegpt4v_row,
        post_load=ensure_allava4v_assets,
    ),
)

for subset in OPC_SUBSETS:
    DATASET_RECIPE_REGISTRY.register(
        name=f"opc_{subset}",
        recipe=DatasetRecipe(
            sources=[{"repo": OPC_REPO, "subset": subset, "split": "train"}],
            processor=process_opc_sft_stage1,
        ),
    )

DATASET_RECIPE_REGISTRY.register(
    name="opc_all",
    recipe=DatasetRecipe(
        sources=[
            {"repo": OPC_REPO, "subset": subset, "split": "train"}
            for subset in OPC_SUBSETS
        ],
        processor=process_opc_sft_stage1,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="gsm8k",
    recipe=DatasetRecipe(
        sources=[{"repo": "openai/gsm8k", "subset": "main", "split": "train"}],
        processor=process_gsm8k_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="hendrycks_math",
    recipe=DatasetRecipe(
        sources=[
            {"repo": "EleutherAI/hendrycks_math", "subset": subject, "split": "train"}
            for subject in HENDRYCKS_MATH_SUBJECTS
        ],
        processor=process_hendrycks_math_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="math_qa",
    recipe=DatasetRecipe(
        sources=[
            {"repo": "allenai/math_qa", "split": "train", "trust_remote_code": True}
        ],
        processor=process_math_qa_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="codealpaca-20k",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "sahil2801/CodeAlpaca-20k",
                "split": "train",
                "trust_remote_code": True,
            }
        ],
        processor=process_codealpaca_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="opencodeinstruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "nvidia/OpenCodeInstruct",
                "split": "train",
                "trust_remote_code": True,
            }
        ],
        processor=process_opencodeinstruct_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="magicoder-evol-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K",
                "split": "train",
                "trust_remote_code": True,
            }
        ],
        processor=process_magicoder_evol_instruct_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="sciq",
    recipe=DatasetRecipe(
        sources=[
            {"repo": "allenai/sciq", "split": "train", "trust_remote_code": True}
        ],
        processor=process_sciq_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="camel",
    recipe=DatasetRecipe(
        sources=[
            {"repo": "camel-ai/biology", "split": "train"},
            {"repo": "camel-ai/chemistry", "split": "train"},
            {"repo": "camel-ai/physics", "split": "train"},
        ],
        processor=process_camel_row,
    ),
)

DATASET_RECIPE_REGISTRY.register(
    name="nebius-llama31-8b-infinity-instruct",
    recipe=DatasetRecipe(
        sources=[
            {
                "repo": "nebius/Llama-3.1-8B-Instruct-Infinity-Instruct-0625",
                "split": "train",
            }
        ],
        processor=process_nebius_infinity_instruct,
        post_load=lambda ds: ds.map(add_index, with_indices=True),
    ),
)
