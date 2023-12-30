from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample = data["train"].select(range(50))

# The attention weights matrix is decomposed from M x M into two lower rank matrices - M x N and N x M
# The rank (r) is a hyperparameter
# A smaller r leads to a simpler low-rank matrix, which results in fewer parameters to learn during adaptation.
# This can lead to faster training and potentially reduced computational requirements.
# However, with a smaller r, the capacity of the low-rank matrix to capture task-specific information decreases.
# This may result in lower adaptation quality, and the model might not perform as well on the new task compared to a higher r.
# other hyperparams: dropout, target_modules

# LoRA config
import peft
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=1,
    lora_alpha=1,  # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=["query_key_value"],
    lora_dropout=0.1, 
    bias="none",  # this specifies if the bias parameter should be trained. 
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(foundation_model, lora_config)
print(peft_model.print_trainable_parameters())

# trainer config
import transformers
from transformers import TrainingArguments, Trainer
import os

output_directory = '.'
training_args = TrainingArguments(
    output_dir=output_directory,
    auto_find_batch_size=True,
    learning_rate= 3e-2,  # higher learning rate than full fine-tuning
    num_train_epochs=5,
    no_cuda=True
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_sample,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# parameter efficient fine-tuning
trainer.train()

peft_model_path = f'{output_directory}/model_peft'
trainer.model.save_pretrained(peft_model_path)

from peft import PeftModel, PeftConfig

loaded_model = PeftModel.from_pretrained(
    foundation_model.to("cpu"), peft_model_path, is_trainable=False)

# inference
inputs = tokenizer("Two things are infinite: ", return_tensors="pt")
outputs = loaded_model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"], 
    max_new_tokens=7, 
    eos_token_id=tokenizer.eos_token_id
    )
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
