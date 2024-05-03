# -*- coding: utf-8 -*-
"""Notebook.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/Nehan757/LangChain_Project/blob/main/Notebook.ipynb
"""

import os

# Access the secret
key = os.getenv('openai_key')

# Use the secret in your code
print("My secret is:", key)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
import socket
import ssl
import dill
import torch


torch.device('cpu')
class Runnable:
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement abstract method")

def load_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

# Responsible for splitting the documents into several chunks

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding
        }
    )


def create_embeddings(chunks, embedding_model, storing_path = "C:\\Users\\nehan\\vectorstore"):


    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

template = """
### System:
You are an respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""

def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def get_response(query, chain):
    response = chain({'query': query})

    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)

from langchain.llms import OpenAI
from langchain import PromptTemplate

class OpenAIWrapper(Runnable):
    def __init__(self, openai_instance):
        self.openai_instance = openai_instance

    def run(self, *args, **kwargs):
        return self.openai_instance.run(*args, **kwargs)

import os
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "key"

# Loading openai
llm = OpenAI()

# Loading the Embedding Model
embed = load_embedding_model(model_path = "all-MiniLM-L6-v2")

# loading and splitting the documents
docs = load_pdf_data(file_path=r"COI.pdf")
documents = split_docs(documents=docs)

# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)



get_response("What if i have done a murder", chain)

# Saving the model

# Serialize the chain components
chain_components = {
    'retriever': retriever,
    'llm': llm,
    'prompt': prompt
}

# Save the chain components
with open('chain_components.pkl', 'wb') as f:
    dill.dump(chain_components, f)

# Load the chain components
with open('chain_components.pkl', 'rb') as f:
    loaded_chain_components = dill.load(f)

# Reconstruct the chain object
retriever2 = loaded_chain_components['retriever']
llm2 = loaded_chain_components['llm']
prompt2 = loaded_chain_components['prompt']

# Create the chain
chain2 = load_qa_chain(retriever, llm, prompt)

# Use the chain
get_response("What is my write of Self-Defence", chain2)
