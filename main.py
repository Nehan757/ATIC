import dill
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate
import torch
import fastapi
from fastapi import Request
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

torch.device('cpu')
print("modules imported")

# Load the chain components
import os
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'chain_components.pkl')
with open(file_path, 'rb') as f:
    loaded_chain_components = dill.load(f)

# Reconstruct the chain object
retriever = loaded_chain_components['retriever']
llm = loaded_chain_components['llm']
prompt = loaded_chain_components['prompt']

# Create the chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# Define a function to get the response
def get_response(query):
    response = chain({'query': query})
    return response['result']

# Initialize FastAPI app
app = FastAPI(
    title="ATIC",
    description="Made by Nehan Tanwar",
    version='1.0.0'
)

# Define Pydantic model for request body
class QueryModel(BaseModel):
    query: str

@app.get("/")
def index():
    return {"message": "Welcome to ATIC"}

@app.post("/prediction_api")
async def predict(query: QueryModel):
    prediction = get_response(query.query)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app)

