import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from dashboard.service.ai import (LLM_CONFIGS,
                                  create_llm, 
                                  load_documents_from_dir, 
                                  query_rag)

from dashboard.service.svn import svn_checkout

from dashboard.service.vectordb import cerca_documenti, connect_weaviate,embedding_docs,create_embedding,ingest_text

load_dotenv()

PREFIX = "RAG_VLLM_"


# --- Recupera le classi esistenti ---
def get_existing_projects():
    client = connect_weaviate()
    # ["--Crea nuovo progetto--"]+
    projects = [""] + [
        col for col in client.collections.list_all() 
        # //if col.startswith(PREFIX)
    ]
    client.close()
    return projects




# --- Streamlit UI ---
st.set_page_config(page_title="RAG Multi-Progetto", layout="wide")
st.title("📚 RAG Multi-Progetto con Weaviate v4")

selected_name = st.selectbox("Choose a model:", [c["name"] for c in LLM_CONFIGS])

# Get config and build LLM
selected_config = next(c for c in LLM_CONFIGS if c["name"] == selected_name)

tab1, tab2,tab3,tab4, tab9 = st.tabs(["IngestionText", "IngestionSvn",  "WEAVIATE UTIL", "Sintesi", "Domande"])

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
    new_project = st.text_input("Inserisci svn url:",value="ENTTBL/MATERIAL/BE/trunk" ,  key="url_svn" )
    username = st.text_input("username:",value="ttornabene" ,  key="username" )
    password = st.text_input("password:",value="" ,  key="password" )
   
    if st.button("ElaboraSvn"):
        st.write(f"Check......: {new_project}")
        dest_path = svn_checkout(
            project=new_project,
            username="ttornabene",
            password="TommasoLuglio2025."
            )
        st.write(f"Checked out to: {dest_path}")
        docs = load_documents_from_dir( dest_path )
        st.write(f"✅ Caricati {len(docs)} documenti da svn url: {dest_path} ")
        client = connect_weaviate()
        embedding_docs(client,docs = docs ,  project=dest_path )
        client.close()
        st.write(f"✅ Ingestione completata")

# --- Tab Ingestione ---
with tab3:
    st.header("WEAIATE UTIL ")
    if st.button("PULISCI WEAVIATE"):
        client = connect_weaviate()
        client.collections.delete_all()
        st.write(f"✅ Weaviate PULITO")
        client.close()



# --- Tab Ingestione ---
with tab4:
    st.header("Sintesi")
    projects = get_existing_projects()
    projects_sistesi = st.selectbox( "Seleziona progetto:", projects,   key="projects_sistesi" )
    query_sintesi = st.text_input("Domanda:","create a description of all  @Entity")
    if st.button("Sintesi"):
       docs = cerca_documenti(projects_sistesi,query_sintesi,top_k=10)
       st.write(docs)
        
# --- Tab Domande ---
with tab9:
    st.header("Fai una domanda")
    projects = get_existing_projects()
    project_query = st.selectbox("Seleziona progetto da interrogare:", projects, index=0 )
    query = st.text_input("Domanda:","create a description of all  @Entity  and the relationship between them in PlantUML format")
    typeQA = st.selectbox(
        "Select chain type:",
        options=["stuff", "map_reduce", "refine"],
        index=0  # default to "stuff"
    )
    top_k= st.number_input("top_k:", min_value=1, max_value=100, value=10, step=1)
            
    if st.button("Cerca"):
        if query.strip() and project_query.strip():
            
            st.write(f"llm model {selected_config} - project {project_query} - typeQA {typeQA} ")
            response, docs = query_rag(query, project_query, selected_config ,typeQA=typeQA ,  top_k=top_k )
            st.subheader("Risposta")
            st.write(response)
            st.subheader("Contesto usato")
            for d in docs:
                st.markdown(f"- {d.page_content[:200]}...")
        else:
            st.warning("Inserisci sia un progetto che una domanda.")