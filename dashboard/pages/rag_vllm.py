import os
import streamlit as st
from dotenv import load_dotenv

from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import weaviate

from dashboard.service.ai import LLM_CONFIGS, connect_weaviate, create_embedding, create_llm, embedding_docs, load_documents_from_dir

load_dotenv()

PREFIX = "RAG_VLLM_"


# --- Recupera le classi esistenti ---
def get_existing_projects():
    client = connect_weaviate()
    # ["--Crea nuovo progetto--"]+
    projects = [""] + [
        col for col in client.collections.list_all() if col.startswith(PREFIX)
    ]
    client.close()
    return projects


# --- Ingestione ---
def ingest_text(text: str, project: str, new_project: str, projects: list[str]):
    client= connect_weaviate()
   
    if new_project != "" and new_project not in projects:
        if new_project.strip():
            project = PREFIX + new_project
            client.collections.create(name=project)
            st.success(f"Progetto '{project}' creato ✅")
    else:
        st.info(f"Usando progetto esistente: {project}")

    st.info(f"WeaviateVectorStore.from_documents: {project}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])

    embeddings = create_embedding()
    WeaviateVectorStore.from_documents(
        client=client,
        documents=docs,
        embedding=embeddings,
        index_name=project,
        text_key="text",
    )

    client.close()
    n = len(docs)
    st.success(f"Ingeriti {n} chunk nel progetto '{project}' ✅")
    return n

# --- Query ---
def query_rag(question: str, project: str, top_k: int = 3):
    client = connect_weaviate()
    
    embeddings = create_embedding()
    store = WeaviateVectorStore(
        client=client,
        index_name=project,
        text_key="text",
        embedding=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    llm = create_llm(selected_config)
    st.write(f"✅ Using model: {selected_config['model']}")

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Rispondi alla domanda basandoti sul contesto:\n\n{context}\n\nDomanda: {question}"
    response = llm.invoke(prompt)

    client.close()
    return response.content, docs

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Multi-Progetto", layout="wide")
st.title("📚 RAG Multi-Progetto con Weaviate v4")

selected_name = st.selectbox("Choose a model:", [c["name"] for c in LLM_CONFIGS])

# Get config and build LLM
selected_config = next(c for c in LLM_CONFIGS if c["name"] == selected_name)

tab1, tab2,tab3, tab9 = st.tabs(["IngestionText", "IngestionSvn",  "WEAVIATE UTIL",  "Domande"])

# --- Tab Ingestione ---
with tab1:
    st.header("Carica documenti")
    projects = get_existing_projects()
    project_ingest = st.selectbox("Seleziona progetto:", projects, key="projects")
    new_project = st.text_input("Inserisci nome nuovo progetto:")
    text = st.text_area("Inserisci testo da indicizzare", height=200)
    if st.button("Ingerisci"):
        if text.strip() and (project_ingest.strip() or new_project.strip()):
            n = ingest_text(text, project_ingest, new_project, projects)

        else:
            st.warning("Inserisci sia un progetto che del testo.")

# --- Tab Ingestione ---
with tab2:
    st.header("Carica svn url")
    projects = get_existing_projects()
    project_ingest = st.selectbox( "Seleziona progetto:", projects,   key="ingest_svn" )
    new_project = st.text_input("Inserisci svn url:",value="volume_data/svn/ENTTBL/MATERIAL/BE/trunk" ,  key="url_svn" )
    # text = st.text_area("Inserisci testo da indicizzare", height=200)
    if st.button("ElaboraSvn"):
        docs = load_documents_from_dir( new_project )
        st.write(f"✅ Caricati {len(docs)} documenti da svn url: {new_project} ")
        client = connect_weaviate()
        embedding_docs(client,docs = docs ,  project=new_project )
        client.close()
        st.write(f"✅ Ingestione completata")
        # if text.strip() and (project_ingest.strip() or new_project.strip()):
        #     n = ingest_text(text, project_ingest, new_project, projects)
        # else:
        #     st.warning("Inserisci sia un progetto che del testo.")

# --- Tab Ingestione ---
with tab3:
    st.header("WEAIATE UTIL ")
    if st.button("PULISCI WEAVIATE"):
        client = connect_weaviate()
        client.collections.delete_all()
        st.write(f"✅ Weaviate PULITO")
        client.close()


# --- Tab Domande ---
with tab9:
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
