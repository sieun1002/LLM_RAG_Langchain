from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
import textwrap
# load INSTRUCTOR_Transformer 
# max_seq_length  512

loader = DirectoryLoader('./data',glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)


instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",model_kwargs={"device":"cuda"})

persist_directory = "EDIYA_ko"

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vectordb.persist()
