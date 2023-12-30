import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import evaluate
import nltk
from nltk.tokenize import sent_tokenize


ds = load_dataset('databricks/databricks-dolly-15k')
model_checkpoint = 'EleutherAI/pythia-70m-deduped'


# configure
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["### End", "### Instruction:", "### Response:\n"]}
)

remove_columns = ["instruction", "response", "context", "category"]


def tokenize(x: dict, max_length: int = 1024) -> dict:
    """
    For a dictionary example of instruction, response, and context a dictionary of input_id and attention mask is returned
    """
    instr = x["instruction"]
    resp = x["response"]
    context = x["context"]

    instr_part = f"### Instruction:\n{instr}"
    context_part = ""
    if context:
        context_part = f"\nInput:\n{context}\n"
    resp_part = f"### Response:\n{resp}"

    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{instr_part}
{context_part}
{resp_part}

### End
"""
    return tokenizer(text, max_length=max_length, truncation=True)

tokenized_dataset = ds.map(
    tokenize, batched=True, remove_columns=remove_columns
)

root_path = '.'
checkpoint_name = "test-trainer-lab"
local_checkpoint_path = os.path.join(root_path, checkpoint_name)
training_args = TrainingArguments(
    local_checkpoint_path,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    optim="adamw_torch",
    report_to=["tensorboard"],
)

checkpoint_name = "test-trainer-lab"

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

TRAINING_SIZE=6000
SEED=42
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)

from datasets import Dataset

shuffled_ds = Dataset.from_dict(tokenized_dataset['train'][:]).shuffle(seed=SEED)
train_ds = Dataset.from_dict(shuffled_ds[:TRAINING_SIZE])
eval_ds = Dataset.from_dict(shuffled_ds[TRAINING_SIZE:])

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

tensorboard_display_dir = f"{local_checkpoint_path}/runs"

trainer.train()

# save model to the local checkpoint
trainer.save_model()
trainer.save_state()

# persist the fine-tuned model to DBFS
final_model_path = f"{root_path}/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

fine_tuned_model = AutoModelForCausalLM.from_pretrained(final_model_path)

def to_prompt(instr: str, max_length: int = 1024) -> dict:
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:
"""
    return tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)


import re


def to_response(prediction):
    decoded = tokenizer.decode(prediction)
    # extract the Response from the decoded sequence
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)
    res = "Failed to find response"
    if m:
        res = m.group(1).strip()
    else:
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            res = m.group(1).strip()
    return res


res = []
for i in range(100):
    instr = ds["train"][i]["instruction"]
    resp = ds["train"][i]["response"]
    inputs = to_prompt(instr)
    pred = fine_tuned_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=128,
    )
    res.append((instr, resp, to_response(pred[0])))

pdf = pd.DataFrame(res, columns=["instruction", "response", "generated"])

# evaluate
nltk.download("punkt")

rouge_score = evaluate.load("rouge")


def compute_rouge_score(generated, reference):
    """
    Compute ROUGE scores on a batch of articles.

    This is a convenience function wrapping Hugging Face `rouge_score`,
    which expects sentences to be separated by newlines.

    :param generated: Summaries (list of strings) produced by the model
    :param reference: Ground-truth summaries (list of strings) for comparison
    """
    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
    )


rouge_scores = compute_rouge_score(pdf['response'], pdf['generated'])
