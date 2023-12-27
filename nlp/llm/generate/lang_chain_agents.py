from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


llm = OpenAI()

# create an agent that collects data from the internet
tools = load_tools(["wikipedia", "serpapi", "python_repl", "terminal"], llm=llm)
dascie = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
dascie.run(
    "Create a dataset (DO NOT try to download one, you MUST create one based on what you find) on the performance of the Mercedes AMG F1 team in 2020 and do some analysis. You need to plot your results."
)

# create an agent that loads data from a file
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

datasci_data_df = pd.read_csv(f"some_path/some_file.csv")
dascie = create_pandas_dataframe_agent(
    llm, datasci_data_df, verbose=True
)
dascie.run("Analyze this data, tell me any interesting trends. Make some pretty plots.")
