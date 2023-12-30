from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "This is a MOOC about large language models, I have only just started, but already"
inputs = tokenizer.encode(prompt, return_tensors='pt')
attention_mask = torch.ones(inputs.shape, dtype=torch.long)
pad_token_id = tokenizer.eos_token_id
print(prompt, end=' ', flush=True)

for _ in range(25):
    outputs = model.generate(
        inputs, max_length=inputs.shape[-1] + 1, do_sample=True,
        pad_token_id=pad_token_id, attention_mask=attention_mask)

    generated_word = tokenizer.decode(outputs[0][-1])
    print(generated_word, end=' ', flush=True)

    # Append the generated token to the input sequence for the next round of generation. We have to add extra dimensions 
    # to the tensor to match the shape of the input tensor (which is 2D: batch size x sequence length).
    inputs = torch.cat([inputs, outputs[0][-1].unsqueeze(0).unsqueeze(0)], dim=-1)

    # Extend the attention mask for the new token. Like before, it should be attended to, so we add a 1.
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
    time.sleep(0.7)
