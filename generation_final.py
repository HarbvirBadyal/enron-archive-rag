import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Final retrieval pipeline
RETRIEVAL_MODULE = "retrieval_final"

# Final generation model
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Retrieval and generation settings
RETRIEVAL_K = 10
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 256
MAX_GENERATION_SECONDS = 60

UNKNOWN_RESPONSE = (
    "I could not find that information in the provided emails."
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_DTYPE = (
    torch.float16
    if DEVICE == "mps"
    else torch.float32
)

print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Retrieval top-k: {RETRIEVAL_K}")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=MODEL_DTYPE,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

if DEVICE == "mps":
    model = model.to("mps")

model.eval()
model.config.use_cache = True

print("Base Qwen loaded.")


def retrieve_for_generation(question):
    """
    Run the final retrieval pipeline in a separate subprocess
    and return the top retrieved chunks for one question.
    """

    retrieval_results_file = Path(
        "generation_retrieval_temp.json"
    )

    retrieval_code = f"""
import json

from {RETRIEVAL_MODULE} import retrieve_chunks

question = {question!r}

chunks = retrieve_chunks(
    question,
    k={RETRIEVAL_K}
)

with open(
    r"{retrieval_results_file}",
    "w"
) as f:
    json.dump(chunks, f)
"""

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "IPython",
            "-c",
            retrieval_code,
        ],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(
            "Retrieval pipeline failed."
        )

    with open(
        retrieval_results_file,
        "r"
    ) as f:
        chunks = json.load(f)

    retrieval_results_file.unlink(
        missing_ok=True
    )

    return chunks


SYSTEM_PROMPT = """
Use the supplied email evidence to answer the question.
You may combine information across multiple pieces of evidence.

Do not use outside knowledge or invent information.

If the answer cannot be determined from the supplied evidence, say:
I could not find that information in the provided emails.

Return:
ANSWER: <answer>
SOURCE: <email_id>
""".strip()


def build_prompt(question, chunks):
    """
    Convert retrieved email chunks into the final generation prompt.
    """

    evidence_blocks = []

    for chunk in chunks:
        evidence_blocks.append(
            f"EMAIL_ID: {chunk['email_id']}\n"
            f"SUBJECT: {chunk.get('subject', '')}\n"
            f"DATE: {chunk.get('date', '')}\n"
            f"EVIDENCE:\n"
            f"{chunk['chunk_text']}"
        )

    evidence = "\n\n---\n\n".join(
        evidence_blocks
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"QUESTION:\n"
        f"{question}\n\n"
        f"EMAIL EVIDENCE:\n"
        f"{evidence}"
    )

    return prompt


def generate_answer(question, chunks):
    """
    Generate a grounded answer from retrieved email chunks.
    """

    prompt = build_prompt(
        question,
        chunks
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    start_time = time.time()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            max_time=MAX_GENERATION_SECONDS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = (
        time.time() - start_time
    )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    time_limit_reached = (
        generation_seconds >= MAX_GENERATION_SECONDS
    )

    valid_response = (
        "ANSWER:" in response
        and "SOURCE:" in response
    )

    if time_limit_reached or not valid_response:
        response = UNKNOWN_RESPONSE

    del inputs
    del output
    del generated_tokens

    gc.collect()

    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "response": response,
        "generation_seconds": generation_seconds,
        "time_limit_reached": time_limit_reached,
    }


import re


def parse_response(response):
    """
    Separate the generated response into answer and source fields.
    """

    if response == UNKNOWN_RESPONSE:
        return {
            "answer": UNKNOWN_RESPONSE,
            "source_email_id": None,
        }

    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=\nSOURCE:|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    source_match = re.search(
        r"SOURCE:\s*(<[^<>]+JavaMail\.evans@thyme>)",
        response,
        flags=re.IGNORECASE,
    )

    answer = (
        answer_match.group(1).strip()
        if answer_match
        else UNKNOWN_RESPONSE
    )

    source_email_id = (
        source_match.group(1)
        if source_match
        else None
    )

    return {
        "answer": answer,
        "source_email_id": source_email_id,
    }


def answer_question(question):
    """
    Run the full RAG pipeline for one user question.
    """

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "source_email_id": None,
            "generation_seconds": 0.0,
            "time_limit_reached": False,
        }

    chunks = retrieve_for_generation(
        question
    )

    if not chunks:
        return {
            "answer": UNKNOWN_RESPONSE,
            "source_email_id": None,
            "generation_seconds": 0.0,
            "time_limit_reached": False,
        }

    generation_result = generate_answer(
        question,
        chunks
    )

    parsed = parse_response(
        generation_result["response"]
    )

    valid_source_ids = {
        chunk["email_id"]
        for chunk in chunks
    }

    if parsed["source_email_id"] not in valid_source_ids:
        parsed["source_email_id"] = None

    return {
        "answer": parsed["answer"],
        "source_email_id": parsed["source_email_id"],
        "generation_seconds": generation_result["generation_seconds"],
        "time_limit_reached": generation_result["time_limit_reached"],
    }


RUN_SANITY_CHECK = globals().get(
    "RUN_SANITY_CHECK",
    True
)

if __name__ == "__main__" and RUN_SANITY_CHECK:
    test_result = answer_question(
        "Who was listed as the authorized director for Enron Credit Limited?"
    )

    print("Answer:")
    print(test_result["answer"])

    print("\nSource:")
    print(test_result["source_email_id"])

    print(
        "\nGeneration time:",
        round(test_result["generation_seconds"], 2),
        "seconds"
    )

    print(
        "\nTime limit reached:",
        test_result["time_limit_reached"]
    )
