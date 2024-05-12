import os
import openai
import langchain
from pinecone import Pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as ps
from langchain_pinecone import PineconeVectorStore
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

import streamlit as st



os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
index_name = 'vartalap'
index = ps.from_existing_index(index_name=index_name, embedding=embeddings, namespace="gita")

llm=OpenAI()
chain=load_qa_chain(llm,chain_type="stuff")

def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results


def retrieve_answers(query):
    query = query + text
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response

st.title('Bhagawad-Gita-Chatbot')
our_query = st.text_input("Ask a question related to Bhagawad Gita :")
text = "if the answer is related to bhagawad gita then it should state chapter number and verse number along with the consice answer"

if our_query:
    st.write(retrieve_answers(our_query))