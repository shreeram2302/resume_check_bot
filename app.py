import streamlit as st
import os
import fitz

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings


Settings.llm = None



def read_resumes_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, file_name)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append(Document(text=text, metadata={"file_name": file_name}))
    return documents


@st.cache_resource
def build_index():
    folder = "resumes"
    documents = read_resumes_from_folder(folder)

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)


    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index


st.set_page_config(page_title="Resume Bot", page_icon="📄")
st.title("📄 Resume Bot (Offline)")
st.write("Put your resumes in the `/resumes` folder and ask questions about skills, experience, etc.")

query = st.text_input("Enter your question (e.g. Who has experience in Python and ML?)")

if query:
    with st.spinner("Searching..."):
        index = build_index()


        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        st.subheader("Answer")
        st.write(response.response)

# python -m streamlit run app.py