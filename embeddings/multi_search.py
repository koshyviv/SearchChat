import faiss
import openai
import os
import pickle
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

# Render Streamlit page
st.set_page_config(page_title='Edge Qs')
st.title("Edge Thoughts")
st.markdown(
    '''
    This mini-app generates answers using OpenAI's GPT-3 based [Davinci model](https://beta.openai.com/docs/models/overview). 
    Embedding have been pre-generated using sentence-transformers/gtr-t5-large, based on Edge content from various McKinsey, Gartner texts, ebooks and Youtube videos from the past year.

    This is a WIP, results will get much better post data cleaning. Since this is limited by 4000 tokens (max), it provides limited results for now

    '''
)

folder = "ebooks"

folders = [["ebooks",0.7],["edge_2",0.3]]

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
INDEX_FILE = "index.pkl"
MAPPING_FILE = "mappings.pkl"
DOCUMENTS_FILE = "documents.pkl"
TOP_K = 12

# openai.api_key = os.environ["open_api_key"]
openai.api_key ="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"

def evaluate_prompt(prompt: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.2,
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

if "model" not in st.session_state:
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st.session_state.model = model

    db = []
    for item in folders:
        index = faiss.read_index(f"./{item[0]}/{INDEX_FILE}")
    
        with open(f"./{item[0]}/{MAPPING_FILE}", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        with open(f"./{item[0]}/{DOCUMENTS_FILE}", "rb") as f:
            docstore = pickle.load(f)
        
        db.append(FAISS(model.embed_query, 
            index, 
            docstore=docstore, 
            index_to_docstore_id=index_to_docstore_id))

    st.session_state.db = db


query = st.text_input("Enter a question")
if query:
    context = ""
    for idx,db in enumerate(st.session_state.db):
        results = db.similarity_search(query, k=int(TOP_K * folders[idx][1]))
        for result in results:
            context += result.page_content
            context += "\n\n- - - - - - -\n\n"
    
    prompt = template.format(context=context, user_question=query)
    with st.expander("Show query"):
        st.write(prompt)

    with st.spinner("Waiting for OpenAI to respond..."):
        response = evaluate_prompt(prompt)
    
    st.write(response)


# import streamlit as st
# from streamlit_chat import message

# """
# ## Text Is All You Need

# A contrarian take on the user interface of the future. 
# """

# if "generated" not in st.session_state:
#     st.session_state.generated = []
#     st.session_state.past = []

# user_input = st.text_input("You", 
#     placeholder="Ask me about Kunnumpurathu Family", key="input")

# if user_input:
#     response = f"I'm sorry. I'm afraid I can't do that: {user_input}" 
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(response)

# if len(st.session_state.generated) > 0:
#     for i in range(len(st.session_state.generated) - 1, -1, -1):
#         message(st.session_state.generated[i], key=str(i))
#         message(st.session_state.past[i], is_user=True, 
#             avatar_style="jdenticon", key=f"{i}_user")