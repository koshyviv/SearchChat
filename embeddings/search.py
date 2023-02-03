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
INDEX_FILE = "index.pkl"
MAPPING_FILE = "mappings.pkl"
DOCUMENTS_FILE = "documents.pkl"
TOP_K = 3

openai.api_key = os.environ["open_api_key"]

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

if "model" not in st.session_state:
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st.session_state.model = model

    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    with open(DOCUMENTS_FILE, "rb") as f:
        docstore = pickle.load(f)
    
    db = FAISS(model.embed_query, 
        index, 
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id)

    st.session_state.db = db

query = st.text_input("Enter a question")
if query:
    results = st.session_state.db.similarity_search(query, k=TOP_K)
    context = ""
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