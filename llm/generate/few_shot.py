from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd


few_shot_pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    max_new_tokens=50
)

eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

prompt =\
"""Given the following set of keywords, generate a tweet using all of them:

[Keywords]: "home, frog, good"
[Tweet]: "In my home the frog feels good"
###
[Keywords]: "Venezuela, banana, cost"
[Tweet]: "In Venezuela you can find banana for almost no cost"
###
[Keywords]: "Beckham, cinema, hamburger"
[Tweet]: "I saw Beckham in the cinema, he was eating a hamburger"
###
[Keywords]: "cheese, cat, socks"
[Tweet]: "It's an interesting fact that the socks I bought for my cat are made of cheese"
###
[Keywords]: "house, warm, winter"
[Tweet]: """

results = few_shot_pipeline(prompt, do_sample=True, eos_token_id=eos_token_id)
print(results[0]["generated_text"])


###
from transformers import T5Tokenizer, T5ForConditionalGeneration

xsum_dataset = load_dataset("xsum", version="1.2.0")
xsum_sample = xsum_dataset["train"].select(range(10))

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Prepare articles for T5, which requires a "summarize: " prefix.
articles = list(map(lambda article: "summarize: " + article, xsum_sample["document"]))

def display_summaries(decoded_summaries: list) -> None:
    """Helper method to display ground-truth and generated summaries side-by-side"""
    results_df = pd.DataFrame(zip(xsum_sample["summary"], decoded_summaries))
    results_df.columns = ["Summary", "Generated"]
    print(results_df)

inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=5,
    min_length=0,
    max_length=10,
    top_k=10,
    top_p=0.4,
    temperature=0.4,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
display_summaries(decoded_summaries)
