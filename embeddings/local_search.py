import faiss
import openai
import os
import pickle
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

"""
## Koshy Questions
"""

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
INDEX_FILE = "edge/index.pkl"
MAPPING_FILE = "edge/mappings.pkl"
DOCUMENTS_FILE = "edge/documents.pkl"
TOP_K = 15

openai.api_key ="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"

def evaluate_prompt(prompt: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

template = PromptTemplate(
        input_variables=["user_question", "context"],
        template="""
    The following is a question from the user: {user_question} 

    Please answer the question using the context below:
    {context}
    """
    )

def get_gpt_response(db, query):
    results = db.similarity_search(query, k=TOP_K)
    context = ""
    for result in results:
        context += result.page_content
        context += "\n\n- - - - - - -\n\n"
        
    prompt = template.format(context=context, user_question=query)
    response = evaluate_prompt(prompt)
    return response

model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
index = faiss.read_index(INDEX_FILE)
with open(MAPPING_FILE, "rb") as f:
    index_to_docstore_id = pickle.load(f)
with open(DOCUMENTS_FILE, "rb") as f:
    docstore = pickle.load(f)
db = FAISS(model.embed_query, 
    index, 
    docstore=docstore, 
    index_to_docstore_id=index_to_docstore_id)


query = "how is edge computing going to impact cloud computing? which key functions will move from cloud to edge?"
get_gpt_response(db, query)