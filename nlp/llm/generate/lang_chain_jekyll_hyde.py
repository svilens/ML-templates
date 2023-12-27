from langchain import PromptTemplate
import numpy as np

jekyll_template = """
You are a social media post commenter, you will respond to the following post with a {sentiment} response. 
Post:" {social_post}"
Comment: 
"""

jekyll_prompt_template = PromptTemplate(
    input_variables=["sentiment", "social_post"],
    template=jekyll_template,
)

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import OpenAI, HuggingFaceHub

from langchain.chains import LLMChain
from better_profanity import profanity

jekyll_llm = OpenAI(model="text-babbage-001")

jekyll_chain = LLMChain(
    llm=jekyll_llm,
    prompt=jekyll_prompt_template,
    output_key="jekyll_said",
    verbose=False,
)

random_sentiment = "nice"
if np.random.rand() < 0.3:
    random_sentiment = "mean"
# We'll also need our social media post:
social_post = "There is a lot more to come about LLMs and the future of NLP looks so, so bright!"

jekyll_said = jekyll_chain.run(
    {"sentiment": random_sentiment, "social_post": social_post}
)
cleaned_jekyll_said = profanity.censor(jekyll_said)

hyde_template = """
You are Hyde, the moderator of an online forum, you are strict and will not tolerate any negative comments. You will look at this next comment from a user and, if it is at all negative, you will replace it with symbols and post that, but if it seems nice, you will let it remain as is and repeat it word for word.
Original comment: {jekyll_said}
Edited comment:
"""

hyde_prompt_template = PromptTemplate(
    input_variables=["jekyll_said"],
    template=hyde_template,
)
hyde_llm = jekyll_llm
hyde_chain = LLMChain(
    llm=hyde_llm, prompt=hyde_prompt_template, verbose=False
)
hyde_says = hyde_chain.run({"jekyll_said": jekyll_said})


# sequential chain
from langchain.chains import SequentialChain

jekyllhyde_chain = SequentialChain(
    chains=[jekyll_chain, hyde_chain],
    input_variables=["sentiment", "social_post"],
    verbose=True,
)
jekyllhyde_chain.run({"sentiment": random_sentiment, "social_post": social_post})
