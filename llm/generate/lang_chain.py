from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


# input data
laptop_reviews = TextLoader("fake_laptop_reviews.txt", encoding="utf8")
document = laptop_reviews.load()

# vector store
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile

tmp_laptop_dir = tempfile.TemporaryDirectory()
tmp_shakespeare_dir = tempfile.TemporaryDirectory()

# First we split the data into manageable chunks to store as vectors.
# There isn't an exact way to do this, more chunks means more detailed context,
# but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(document)

# Now we'll create embeddings for our document so we can store it in a vector store
# and feed the data into an LLM. We'll use the sentence-transformers model for
# our embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# create index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory=tmp_laptop_dir.name
)

# create retrieval chain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

retriever = chromadb_index.as_retriever()

hf_llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text2text-generation",
    model_kwargs={
        "temperature": 0,
        "max_length": 128
    },
)

chain_type = "stuff"  # stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, chain_type=chain_type, retriever=retriever
)

# retrieve
laptop_name = laptop_qa.run("What is the full name of the laptop?")
laptop_features = laptop_qa.run("What are some of the laptop's features?")
laptop_reviews = laptop_qa.run("What is the general sentiment of the reviews?")
