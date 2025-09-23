import os
from dashboard.service.ai import connect_weaviate
import streamlit as st
from dotenv import load_dotenv

import weaviate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Config ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Recupera le classi esistenti ---
def get_existing_projects():
    client = connect_weaviate()
    # ["--Crea nuovo progetto--"]+
    projects =  [""] + [  col for col in client.collections.list_all() ]
    client.close()
    return projects

# --- Ingestione ---
def ingest_text(text: str, project: str,new_project: str, projects:list[str] ):
    client = connect_weaviate()
    embeddings = OpenAIEmbeddings()
    if new_project != "" and new_project not in projects:
        if new_project.strip():
            client.collections.create( name=new_project)
            st.success(f"Progetto '{new_project}' creato ✅")
            project = new_project
    else:
        st.info(f"Usando progetto esistente: {project}")

    st.info(f"WeaviateVectorStore.from_documents: {project}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])

    WeaviateVectorStore.from_documents(
        client=client,
        documents=docs,
        embedding=embeddings,
        index_name=project,
        text_key="text",
    )

    client.close()
    n =  len(docs)
    st.success(f"Ingeriti {n} chunk nel progetto '{project}' ✅")
    return n

# --- Query ---
def query_rag(question: str, project: str, top_k: int = 3):
    client = connect_weaviate()
    embeddings = OpenAIEmbeddings()

    store = WeaviateVectorStore(
        client=client,
        index_name=project,
        text_key="text",
        embedding=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Rispondi alla domanda basandoti sul contesto:\n\n{context}\n\nDomanda: {question}"
    response = llm.invoke(prompt)

    client.close()
    return response.content, docs

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Multi-Progetto", layout="wide")
st.title("📚 RAG Multi-Progetto con Weaviate v4")

tab1, tab2 = st.tabs(["Ingestione", "Domande"])

# --- Tab Ingestione ---
with tab1:
    st.header("Carica documenti")
    projects = get_existing_projects()
    project_ingest = st.selectbox("Seleziona progetto:", projects)
    new_project = st.text_input("Inserisci nome nuovo progetto:")
     
    text = st.text_area("Inserisci testo da indicizzare", height=200)
    if st.button("Ingerisci"):
        if text.strip() and ( project_ingest.strip() or new_project.strip() ) :
            n = ingest_text(text, project_ingest, new_project, projects )
          
        else:
            st.warning("Inserisci sia un progetto che del testo.")

# --- Tab Domande ---
with tab2:
    st.header("Fai una domanda")
    projects = get_existing_projects()
    project_query = st.selectbox("Seleziona progetto da interrogare:", projects)
    query = st.text_input("Domanda:")
    if st.button("Cerca"):
        if query.strip() and project_query.strip():
            answer, docs = query_rag(query, project_query)
            st.subheader("Risposta")
            st.write(answer)

            st.subheader("Contesto usato")
            for d in docs:
                st.markdown(f"- {d.page_content[:200]}...")
        else:
            st.warning("Inserisci sia un progetto che una domanda.")
            
# def ingest_text(text: str, project: str):
#     client = connect_weaviate()
#     embeddings = OpenAIEmbeddings()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_documents([Document(page_content=text)])

#     vectorstore = WeaviateVectorStore.from_documents(
#         client=client,
#         documents=docs,
#         embedding=embeddings,
#         index_name=f"RAG_{project}",   # 🔹 ogni progetto ha la sua collection
#         text_key="text",
#     )

#     client.close()
#     return len(docs)


# def query_rag(question: str, project: str, top_k: int = 3):
#     client = connect_weaviate()
#     embeddings = OpenAIEmbeddings()

#     store = WeaviateVectorStore(
#         client=client,
#         index_name=f"RAG_{project}",   # 🔹 usa la collection del progetto scelto
#         text_key="text",
#         embedding=embeddings,
#     )
#     retriever = store.as_retriever(search_kwargs={"k": top_k})
#     docs = retriever.get_relevant_documents(question)

#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#     context = "\n\n".join([d.page_content for d in docs])
#     prompt = f"Rispondi alla domanda basandoti sul contesto:\n\n{context}\n\nDomanda: {question}"
#     response = llm.invoke(prompt)

#     client.close()
#     return response.content, docs


# # --- UI Streamlit ---
# st.title("📚 RAG Multi-Progetto con Weaviate v4")

# tab1, tab2 = st.tabs(["Ingestione", "Domande"])

# with tab1:
#     st.header("Carica documenti")
#     project = st.text_input("Nome progetto:", value="default")
#     text = st.text_area("Inserisci testo da indicizzare", height=200)
#     if st.button("Ingerisci"):
#         if text.strip() and project.strip():
#             n = ingest_text(text, project)
#             st.success(f"Ingeriti {n} chunk in progetto '{project}' ✅")
#         else:
#             st.warning("Inserisci sia un progetto che del testo.")

# with tab2:
#     st.header("Fai una domanda")
#     project = st.text_input("Nome progetto da interrogare:", value="default")
#     query = st.text_input("Domanda:")
#     if st.button("Cerca"):
#         if query.strip() and project.strip():
#             answer, docs = query_rag(query, project)
#             st.subheader("Risposta")
#             st.write(answer)

#             st.subheader("Contesto usato")
#             for d in docs:
#                 st.markdown(f"- {d.page_content[:200]}...")
#         else:
#             st.warning("Inserisci sia un progetto che una domanda.")