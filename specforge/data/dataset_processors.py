import hashlib
import os
import random
import subprocess
from typing import Dict, Tuple

from datasets import config


ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def get_vlm_asset_dir(dataset_name):
    if dataset_name == "sharegpt4v":
        raise ValueError("Downloading 'sharegpt4v' is not supported.")
    elif dataset_name == "allava4v":
        return os.path.join(
            config.HF_DATASETS_CACHE, "FreedomIntelligence", "ALLaVA"
        )
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not a supported VLM dataset for download."
        )


def download_vlm_dataset(dataset_name: str) -> None:
    """Download VLM's dataset such as sharegpt4v and allava4v"""
    if dataset_name == "sharegpt4v":
        raise Exception("Don't Support Download sharegpt4v.")
    elif dataset_name == "allava4v":
        cache_dir = get_vlm_asset_dir(dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            "download_laion.sh",
        )
        os.chmod(script_path, 0o755)
        if not os.path.exists(
            os.path.join(cache_dir, "allava_laion", "image_chunks", "images_0.zip")
        ):
            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Download image dataset failed: {result.stderr}")
            print("##### allava4v dataset Download Complete #####")
        else:
            print("##### allava4v dataset has existed.")
    else:
        raise Exception(f"Don't support {dataset_name}")


def identity_processor(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    return row, 0


def process_ultrachat_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the ultrachat dataset.

    The function expects a row with the following schema:
    "messages": [
        {
            "role": "user" | "assistant",
            "content": str
        }
    ]
    """
    conversations = row["messages"]
    formatted_conversations = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        assert role in ["user", "assistant"]
        formatted_conversations.append({"role": role, "content": content})
    row = {"id": row["prompt_id"], "conversations": formatted_conversations}
    return row, 0


def process_sharegpt_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """
    sharegpt dataset schema:
    {
        "conversations": [
            {
                "from": <system|human|gpt>,
                "value": <message>,
            },
            ...
        ]
    }
    """
    conversations = row["conversations"]
    formatted_conversations = []
    skipped_count = 0
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            skipped_count += 1
            continue
        new_role = ROLE_MAPPING[message["from"]]
        content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "conversations": formatted_conversations}
    return row, skipped_count


def process_sharegpt4v_row(row: Dict, dataset_name: str = None) -> Tuple[Dict | None, int]:
    """
    sharegpt4v dataset schema:
    {
        "id": str,
        "image": str,  # path to the image
        "conversations": [
            {
                "from": <human|gpt>,
                "value": <message>,
            },
            ...
        ]
    }
    """
    cache_dir = get_vlm_asset_dir(dataset_name)
    conversations = row["conversations"]
    image = os.path.join(cache_dir, row["image"])
    if not os.path.exists(image):
        print(f"Image path {image} does not exist, skipping this sample.")
        return None, 0
    formatted_conversations = []
    skipped_count = 0
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            skipped_count += 1
            continue
        new_role = ROLE_MAPPING[message["from"]]
        if new_role == "user":
            text_content = message["value"].replace("<image>\n", "")
            content = text_content
        else:
            content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "image": image, "conversations": formatted_conversations}
    return row, skipped_count


def process_nebius_infinity_instruct(
    row: Dict, dataset_name: str = None
) -> Tuple[Dict, int]:
    conversation = row["conversation"][0]
    generated_message = row["generated_message"]
    formatted_conversations = [
        {"role": "user", "content": conversation["content"]},
        {"role": "assistant", "content": generated_message["content"]},
    ]
    row = {"id": str(row["id"]), "conversations": formatted_conversations}
    return row, 0


def process_opc_sft_stage1(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_codealpaca_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the CodeAlpaca-20k dataset.

    The function expects a row with the following schema:
    {
        "instruction": str,
        "input": str,
        "output": str
    }
    """
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_opencodeinstruct_row(
    row: Dict, dataset_name: str = None
) -> Tuple[Dict, int]:
    """Process a row from the nvidia/OpenCodeInstruct dataset.

    The function expects a row with the following schema:
    {
        "id": str,
        "input": str,
        "output": str,
        "domain": str,
        "generation_algorithm": str,
        "llm_judgement": str,
        "unit_tests": str,
        "tests_execution_status": str,
        "average_test_score": float
    }
    """
    row_id = row.get("id")
    if row_id is None:
        row_id = hashlib.md5((row["input"] + row["output"]).encode()).hexdigest()

    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_magicoder_evol_instruct_row(
    row: Dict, dataset_name: str = None
) -> Tuple[Dict, int]:
    """Process a row from the ise-uiuc/Magicoder-Evol-Instruct-110K dataset.

    The function expects a row with the following schema:
    {
        "instruction": str,
        "response": str
    }
    """
    row_id = hashlib.md5((row["instruction"] + row["response"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ],
    }
    return processed_row, 0


def process_gsm8k_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the gsm8k dataset.

    The function expects a row with the following schema:
    {
        "question": str,
        "answer": str
    }
    """
    row_id = hashlib.md5((row["question"] + row["answer"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
    }
    return processed_row, 0


def process_hendrycks_math_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the hendrycks_math dataset.

    The function expects a row with the following schema:
    {
        "problem": str,
        "solution": str,
        "level": str,
        "type": str
    }
    """
    row_id = hashlib.md5((row["problem"] + row["solution"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ],
    }
    return processed_row, 0


def process_math_qa_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the allenai/math_qa dataset.

    The function expects a row with the following schema:
    {
        "Problem": str,
        "Rationale": str,
        "options": str,  # format: "a) option1 b) option2 c) option3 d) option4"
        "correct": str,
        "annotated_formula": str,
        "linear_formula": str,
        "category": str
    }
    """
    problem = row["Problem"]
    options = row["options"]
    user_content = f"{problem}\n{options}"
    rationale = row["Rationale"]

    row_id = hashlib.md5((user_content + rationale).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rationale},
        ],
    }
    return processed_row, 0


def process_sciq_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the allenai/sciq dataset.

    The function expects a row with the following schema:
    {
        "question": str,
        "distractor3": str,
        "distractor1": str,
        "distractor2": str,
        "correct_answer": str,
        "support": str
    }
    """
    question = row["question"]
    correct_answer = row["correct_answer"]
    distractor1 = row["distractor1"]
    distractor2 = row["distractor2"]
    distractor3 = row["distractor3"]
    support = row["support"]

    answers_list = [distractor3, distractor1, distractor2, correct_answer]
    random.shuffle(answers_list)

    labels = ["a", "b", "c", "d"]
    options_list = [(labels[i], answers_list[i]) for i in range(4)]

    correct_label = None
    for label, answer in options_list:
        if answer == correct_answer:
            correct_label = label
            break

    options_text = "\n".join([f"{label}) {answer}" for label, answer in options_list])
    user_content = f"{question}\n{options_text}"
    assistant_content = f"{support}\nanswer: {correct_label}) {correct_answer}"

    row_id = hashlib.md5((user_content + assistant_content).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }
    return processed_row, 0


def process_camel_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the camel-ai dataset.

    The function expects a row with the following schema:
    {
        "message_1": str,  # user message
        "message_2": str,  # assistant message
    }
    """
    message_1 = row["message_1"]
    message_2 = row["message_2"]

    row_id = hashlib.md5((message_1 + message_2).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": message_1},
            {"role": "assistant", "content": message_2},
        ],
    }
    return processed_row, 0


def add_index(row, idx) -> Dict:
    row["id"] = idx
    return row


def ensure_allava4v_assets(ds):
    download_vlm_dataset("allava4v")
    return ds


def raise_unsupported_sharegpt4v(ds):
    raise NotImplementedError("sharegpt4v is not supported now.")
