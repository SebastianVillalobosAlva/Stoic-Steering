"""LoRA: prep, train (Pass B / Colab), and merge (Pass A, Stage 4).

Pass A only *merges the frozen clean adapters* — it never retrains them.
The two rules that kill the old repo's bugs live here:

- **Fresh base per adapter.** Each merge loads its own pristine base; adapters
  can never stack, and the caller's baseline model is never touched. The
  caller re-checks base integrity (0.542 → 0.542, drift 0) around the merges.
- **Tokenizer comes from the BASE model, never the adapter folder** (the
  tokenizer.json shipped inside each adapter dir is redundant and ignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from stoic.config import DEVICE, DTYPE, GENERATED_DIR, MODEL_NAME

LORA_CONFIG = dict(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def merge_adapter(
    adapter_dir: str | Path,
    model_name: str = MODEL_NAME,
    dtype: torch.dtype = DTYPE,
    device: str = DEVICE,
):
    """Merge one LoRA adapter onto a FRESH base and return the merged model.

    Loads a new base every call — no adapter stacking is possible, and no
    previously loaded model is modified. Use the BASE tokenizer with the
    returned model.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"  fresh base + merge {Path(adapter_dir).name} ...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.config.use_cache = True
    merged.eval()
    return merged


# --- Pass B: training data prep + training (Colab T4) ---------------------

def prep_jsonl(pairs_file: str | Path, author: str) -> Path:
    """Write {author}_train.jsonl (one {"text": stoic_text} per line) from a
    contrastive-pairs file. Output goes to data/generated/, never reference/."""
    with open(pairs_file) as f:
        data = json.load(f)
    pairs = data["pairs"] if isinstance(data, dict) else data

    out_dir = GENERATED_DIR / "lora_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{author}_train.jsonl"
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps({"text": p["stoic_text"]}) + "\n")
    print(f"✓ {len(pairs)} training examples -> {out}")
    return out


def train(
    train_jsonl: str | Path,
    output_dir: str | Path,
    model_name: str = MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    device: str = "cuda",
):
    """Train one LoRA adapter (canonical recipe: r=8, α=32, q_proj+v_proj,
    3 epochs). Pass B; intended for a Colab T4, not the local CPU."""
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to(device)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(train_jsonl))
    tokenized = dataset.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_total_limit=2,
            fp16=False,
            report_to="none",
        ),
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    print(f"✓ adapter saved to {output_dir}")
    return Path(output_dir)
