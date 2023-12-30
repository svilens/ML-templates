import pandas as pd


# input data
data_pdf = pd.read_parquet("data.parquet")
data_pdf["full_text"] = data_pdf.apply(
    lambda row: f"""Title: {row["Title"]}
                Abstract:  {row["Abstract"]}""".strip(),
    axis=1,
)
texts = data_pdf["full_text"].to_list()


# db client
import chromadb
from chromadb.config import Settings


chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="some_path"
    )
)

# create collection
collection_name = "my_talks"

if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)
else:
    print(f"Creating collection: '{collection_name}'")
    talks_collection = chroma_client.create_collection(name=collection_name)

talks_collection.add(
    documents=texts,
    ids=[f"id{x}" for x in range(len(texts))]
)

# query
import json

results = talks_collection.query(
    query_texts="dogs",
    n_results=10
)

print(json.dumps(results, indent=4))


# add a language model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation", model=lm_model, tokenizer=tokenizer, max_new_tokens=512,
    device_map="auto", handle_long_generation="hole"
)

# prompt engineering
question = "Summarize: The latest 3 articles related to dogs"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
prompt_template = f"Relevant context: {context}\n\n The user's question: {question}"

# submit query
lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])

