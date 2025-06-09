"""
Test a fine-tuned Unsloth/LLM review-label model on
1 000 random samples taken *after* the first 20 000 rows
of the original CSV.  
Assumes your fine-tuned model + tokenizer are already on disk
(e.g. in ./ft_model) and CUDA is available.

pip install datasets pandas numpy torch transformers unsloth
"""

import re
import random
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel          # noqa: Unsloth ≥0.6
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# 1.  Load rows AFTER the first 20 000 and keep 1 000 random samples
# ---------------------------------------------------------------------
CSV_PATH       = "fake reviews dataset.csv"
START_ROW      = 20_000       # zero-based slice point
N_SAMPLES      = 1_000
RNG_SEED       = 42

df = pd.read_csv(CSV_PATH, skiprows=range(1, START_ROW + 1))  # skip header + 20 000 rows
df = df[["text_", "label"]].dropna()
df = df[df["label"].isin(["CG", "OR"])]

# Map CG→1, OR→0  (ground-truth)
df["true_label"] = (df["label"] == "CG").astype(int)

# If fewer than N_SAMPLES rows, use all of them
sample_df = df.sample(
    n=min(N_SAMPLES, len(df)),
    random_state=RNG_SEED
).reset_index(drop=True)

# ---------------------------------------------------------------------
# 2.  Load tokenizer / model and prepare the chat template
# ---------------------------------------------------------------------
MODEL_DIR = "./lora_model"      # path to your fine-tuned model folder

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

# tell Unsloth to use the Qwen-2.5 chat schema (same one used in fine-tuning)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = (
    "You are ReviewLabeler, an expert at distinguishing computer-generated "
    "reviews (label 1) from authentic human reviews (label 0)."
)

# ---------------------------------------------------------------------
# 3.  Helper: run one review through the model and extract predicted label
# ---------------------------------------------------------------------
LABEL_RE = re.compile(r"label\s*:\s*([01])", re.I)

@torch.inference_mode()
def predict_label(review_text: str) -> int:
    """Return 0 or 1 extracted from model's answer (defaults to −1 on failure)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"The review you need to label carefully: {review_text}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.8,
        top_p=0.9,
    )
    completion = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    match = LABEL_RE.search(completion)
    if match:
        return int(match.group(1))
    # Fallback: if “computer generated” / “human written” appear alone
    if "computer generated" in completion.lower():
        return 1
    if "human written" in completion.lower():
        return 0
    return -1   # could not parse


# ---------------------------------------------------------------------
# 4.  Run evaluation — with live progress
# ---------------------------------------------------------------------
from tqdm.auto import tqdm                      # progress bar

y_true, y_pred, failures = [], [], 0
progress_bar = tqdm(sample_df.itertuples(index=False), total=len(sample_df),
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

for row in progress_bar:
    true_lbl = row.true_label
    pred_lbl = predict_label(row.text_)
    if pred_lbl == -1:                          # could not parse model output
        failures += 1
        progress_bar.set_postfix_str("⚠ parse-fail")
        continue

    y_true.append(true_lbl)
    y_pred.append(pred_lbl)

    # live accuracy so far
    acc_so_far = (np.array(y_true) == np.array(y_pred)).mean()
    progress_bar.set_postfix(acc=acc_so_far, fails=failures)

# -----------------  Final metrics -----------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = (y_true == y_pred).mean()
precision = (y_pred[y_true == 1] == 1).mean() if (y_pred == 1).any() else 0.0
recall    = (y_pred[y_true == 1] == 1).sum() / max((y_true == 1).sum(), 1)
f1        = (2 * precision * recall) / max((precision + recall), 1e-8)

print("\n=== FINAL RESULTS ===")
print(f"Samples evaluated  : {len(y_true)} / {len(sample_df)}")
print(f"Parse failures     : {failures}")
print(f"Accuracy           : {accuracy:6.3%}")
print(f"Precision (label 1): {precision:6.3%}")
print(f"Recall    (label 1): {recall:6.3%}")
print(f"F1         (label 1): {f1:6.3%}")
