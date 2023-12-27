import shap
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("gpt2")

model.config.is_decoder = True
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0,  # to turn off randomness
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}

input_sentence = ["Sunny days are the best days to go to the beach. So"]

explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(input_sentence)

# we can check the contribution of each input token towards the output token.
# "Red" means positive contribution whereas "blue" means negative indication.
# The color intensity indicates the strength of the contribution. 
shap.plots.text(shap_values)

# The plot below shows which input tokens contributes most towards the output token `looking`. 
shap.plots.bar(shap_values[0, :, "looking"])
