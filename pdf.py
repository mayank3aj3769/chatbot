import os 
from dotenv import load_dotenv
from pathlib import Path

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

from pinecone import Pinecone, ServerlessSpec

from tqdm.autonotebook import tqdm


dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY


## extract data from the pdf

def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents=loader.load() 

    return documents

#Create text chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20) ## overlapp b/w embedddings
    text_chunks=text_splitter.split_documents(extracted_data)

    return text_chunks
    
