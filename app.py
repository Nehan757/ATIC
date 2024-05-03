import streamlit as st
import dill
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate
import torch

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
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True,
                                    chain_type_kwargs={'prompt': prompt})


# Define a function to get the response
def get_response(query):
    response = chain({'query': query})
    return response['result']


# Streamlit app
def main():
    st.title('Ask the Indian Constitution')
    question = st.text_input('Enter your question below')

    if st.button('Get Response'):
        response = get_response(question)
        st.subheader('Response:')
        st.write(response)




    st.markdown('<p style="color:blue; text-align:right;">This AI-tool was developed by Nehan Tanwar</p>', unsafe_allow_html=True)



if __name__ == '__main__':
    main()

