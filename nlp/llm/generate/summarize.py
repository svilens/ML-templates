from datasets import load_dataset
from transformers import pipeline
import pandas as pd


xsum_dataset = load_dataset("xsum", version="1.2.0")
xsum_sample = xsum_dataset["train"].select(range(10))

summarizer = pipeline(
  "summarization",
   model="Falconsai/text_summarization"
)

summarization_results = summarizer(xsum_sample['document'], max_length=100, min_length=20, do_sample=False)
result = pd.DataFrame.from_dict(summarization_results).rename(
    {"summary_text": "generated_summary"}, axis=1).join(
        pd.DataFrame.from_dict(xsum_sample))[["generated_summary", "summary", "document"]]
