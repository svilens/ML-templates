from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd


jpn_dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-jpn_Hani")
jpn_sample = (
    jpn_dataset["test"]
    .select(range(10))
    .rename_column("sourceString", "English")
    .rename_column("targetString", "Japanese")
    .remove_columns(["sourceLang", "targetlang"])
)

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translation_pipeline = pipeline(
  "translation",
  model=model,
  tokenizer=tokenizer,
  src_lang="jpn_Jpan",
  tgt_lang="eng_Latn"
)

translation_results = translation_pipeline(jpn_sample["Japanese"], max_length=200)
translation_results_df = pd.DataFrame.from_dict(translation_results).join(jpn_sample.to_pandas())
