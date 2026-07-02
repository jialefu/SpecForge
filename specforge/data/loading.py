import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk


def _json_compatible(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _normalize_message(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None

    role = message.get("role")
    content = message.get("content")
    if role is None or content is None:
        return None

    normalized = {
        "role": role,
        "content": _json_compatible(content),
    }
    if message.get("tool_calls") is not None:
        normalized["tool_calls"] = _json_compatible(message["tool_calls"])
    return normalized


def _normalize_row(
    row: Dict[str, Any],
    *,
    is_preformatted: bool = False,
    is_vlm: bool = False,
) -> Dict[str, Any]:
    if is_preformatted:
        normalized = {"text": row.get("text", "")}
    else:
        conversations = row.get("conversations") or []
        normalized = {
            "conversations": [
                message
                for message in (_normalize_message(item) for item in conversations)
                if message is not None
            ]
        }

    if "id" in row:
        normalized["id"] = row["id"]

    if not is_preformatted and row.get("tools") is not None:
        normalized["tools"] = _json_compatible(row["tools"])

    if is_vlm and row.get("image") is not None:
        normalized["image"] = row["image"]

    return normalized


def _iter_jsonl_rows(
    file_paths: Iterable[str],
    *,
    is_preformatted: bool,
    is_vlm: bool,
):
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _normalize_row(
                    json.loads(line),
                    is_preformatted=is_preformatted,
                    is_vlm=is_vlm,
                )


def _select_split(dataset, split: str):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(
                f"Expected split '{split}', but found splits: {list(dataset.keys())}"
            )
        return dataset[split]
    return dataset


def _local_data_files(path: Path, suffixes: List[str]) -> List[str]:
    return sorted(
        str(file_path)
        for suffix in suffixes
        for file_path in path.rglob(f"*{suffix}")
        if file_path.is_file()
    )


def _load_local_dataset(path: Path, split: str, is_preformatted: bool, is_vlm: bool):
    suffix = path.suffix.lower()
    if path.is_file():
        if suffix == ".jsonl":
            return Dataset.from_generator(
                _iter_jsonl_rows,
                gen_kwargs={
                    "file_paths": [str(path)],
                    "is_preformatted": is_preformatted,
                    "is_vlm": is_vlm,
                },
            )
        if suffix == ".json":
            return load_dataset("json", data_files=str(path), split=split)
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split=split)
        raise ValueError(f"Unsupported dataset file extension: {suffix}")

    if not path.is_dir():
        return None

    try:
        return _select_split(load_from_disk(str(path)), split)
    except (FileNotFoundError, ValueError):
        pass

    parquet_files = _local_data_files(path, [".parquet"])
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, split=split)

    jsonl_files = _local_data_files(path, [".jsonl"])
    if jsonl_files:
        return Dataset.from_generator(
            _iter_jsonl_rows,
            gen_kwargs={
                "file_paths": jsonl_files,
                "is_preformatted": is_preformatted,
                "is_vlm": is_vlm,
            },
        )

    json_files = _local_data_files(path, [".json"])
    if json_files:
        return load_dataset("json", data_files=json_files, split=split)

    raise ValueError(f"No supported dataset files found under {path}")


def _load_remote_dataset(data_path: str, split: str):
    warnings.warn(
        (
            f"Could not find local dataset path '{data_path}'. Trying to load it as "
            "a Hugging Face dataset id. For large training datasets, download the "
            "dataset locally and pass that local path instead."
        ),
        UserWarning,
        stacklevel=2,
    )
    try:
        return load_dataset(data_path, split=split)
    except Exception as exc:
        raise ValueError(
            (
                f"Could not find local dataset path '{data_path}', and loading it "
                "as a Hugging Face dataset id also failed. If this should be a "
                "local dataset, check the path. Otherwise pass a valid Hugging "
                "Face dataset repo id, or download the dataset with "
                "`huggingface-cli download ... --repo-type dataset --local-dir "
                "<path>` and use that local path."
            )
        ) from exc


def _normalize_dataset(dataset, *, is_preformatted: bool, is_vlm: bool):
    if isinstance(dataset, Dataset):
        columns = dataset.column_names
        features = _normalized_features(dataset, is_preformatted, is_vlm)
        return dataset.map(
            lambda row: _normalize_row(
                row,
                is_preformatted=is_preformatted,
                is_vlm=is_vlm,
            ),
            remove_columns=columns,
            features=features,
        )
    return dataset


def _has_conversation_field(dataset: Dataset, field_name: str) -> bool:
    return field_name in str(dataset.features.get("conversations", ""))


def _normalized_features(
    dataset: Dataset,
    is_preformatted: bool,
    is_vlm: bool,
) -> Features:
    fields = {}
    if is_preformatted:
        fields["text"] = Value("string")
    else:
        message = {
            "role": Value("string"),
            "content": Value("string"),
        }
        if _has_conversation_field(dataset, "tool_calls"):
            message["tool_calls"] = Value("string")
        fields["conversations"] = [message]

    if "id" in dataset.column_names:
        fields["id"] = Value("string")
    if not is_preformatted and "tools" in dataset.column_names:
        fields["tools"] = Value("string")
    if is_vlm and "image" in dataset.column_names:
        fields["image"] = dataset.features["image"]
    return Features(fields)


def load_conversation_dataset(
    data_path: str,
    *,
    split: str = "train",
    is_preformatted: bool = False,
    is_vlm: bool = False,
):
    """Load a conversation dataset from local files or an HF dataset id.

    Local files and directories are preferred. Supported local formats are JSONL,
    JSON, Parquet, and datasets saved with ``Dataset.save_to_disk``. Remote HF
    dataset ids are accepted as a compatibility fallback.
    """

    path = Path(data_path)
    if path.exists():
        dataset = _load_local_dataset(path, split, is_preformatted, is_vlm)
    else:
        dataset = _load_remote_dataset(data_path, split)

    dataset = _select_split(dataset, split)
    return _normalize_dataset(
        dataset,
        is_preformatted=is_preformatted,
        is_vlm=is_vlm,
    )
