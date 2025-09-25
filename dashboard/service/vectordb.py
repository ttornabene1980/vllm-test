import os
from typing import List
from pandas import DataFrame
import weaviate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.client import WeaviateClient
from langchain_weaviate import WeaviateVectorStore
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler

from dashboard.service.language_splitter import split_code_str, split_code_with_metadata
 
 
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "xps")

# --- Connessione Weaviate 4 ---
def connect_weaviate():
    return weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=8080,
        grpc_secure=False,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=50051,
    )



def load_documents(files: List[str], project: str, chunk_size:int ) -> List[Document]:
    project2 = project.replace("/", "_").replace(":", "_").replace(".", "_")
    project2 = project2.replace("__volume_data_svn_", "SVN_").upper()
    print(f"Project:{project} -> {project2}")
    loaded_docs = []
    for path in files:
        docs = split_code_with_metadata(path, chunk_size=chunk_size, chunk_overlap=50)
        loaded_docs.extend(docs)
 
    return loaded_docs


def load_pandas(df: DataFrame, project: str, chunk_size:int ) -> List[Document]:
    project2 = project.replace("/", "_").replace(":", "_").replace(".", "_")
    project2 = project2.replace("__volume_data_svn_", "SVN_").upper()
    print(f"Project:{project} -> {project2}")
    loaded_docs = []
    print( df.head() )
    for index, row in df.iterrows():
        print(f"Row {index}: ")
        docs = split_code_str(row['aggregated_summaries'] , row['path'] , chunk_size=chunk_size, chunk_overlap=50)
        loaded_docs.extend(docs)
 
    return loaded_docs



def embedding_docs(client: WeaviateClient, docs, project: str):
    project2 = project.replace("/", "_").replace(":", "_").replace(".", "_")
    project2 = project2.replace("__volume_data_svn_", "SVN_").upper()

    print(f"Project:{project} -> {project2}")
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=4096, 
    #     chunk_overlap=200  # adjust as needed (500–2000 is typical)
    # )

    loaded_docs = []
    for path in docs:
        # print(f"Loading {path}... {os.path.isfile(path)}")
        if os.path.isfile(path):
            docs = split_code_with_metadata(path,chunk_size=2000,chunk_overlap=100 )
            loaded_docs.extend(docs)
            # with open(path, "r", encoding="utf-8", errors="ignore") as f:
            #     text = f.read()
            #     # print(f"Loaded {len(text)} characters from {path}") aggiungere extrazione language java,javascript, spiegazione generatvo con llm
            #     doc = Document(page_content=text, metadata={"source": path})
            #     chunks = splitter.split_documents([doc])
                
            #     loaded_docs.extend(chunks)
        else:
            print(f"Path {path} is not a file, skipping.")

    print(f"Loaded {len(loaded_docs)} splitter documents.")

    embeddings = create_embedding()
    print(f"embeddings {embeddings}")
    WeaviateVectorStore.from_documents(
        client=client,
        documents=loaded_docs,
        embedding=embeddings,
        index_name=project2,
        text_key="text",
    )
    print(f"WeaviateVectorStore {len(loaded_docs)} splitter documents.")
    return loaded_docs


def create_embedding(config=None) -> HuggingFaceEmbeddings:
    # if config["type"] == "openai":
    #     return OpenAIEmbeddings(model=config["model"])
    # elif config["type"] == "huggingface":
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# elif config["type"] == "gpt4all":
#     return GPT4AllEmbeddings(model_path=config["model"])
# else:
#     raise ValueError(f"Unknown embedding type: {config['type']}")




# --- Ingestione ---
def ingest_text(text: str, project: str, new_project: str, projects: list[str]):
    client = connect_weaviate()

    if new_project != "" and new_project not in projects:
        if new_project.strip():
            project = new_project
            client.collections.create(name=project)
            print(f"Progetto '{project}' creato ✅")
    else:
        print(f"Usando progetto esistente: {project}")

    print(f"WeaviateVectorStore.from_documents: {project}")
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
    print(f"Ingeriti {n} chunk nel progetto '{project}' ✅")
    return n

def cerca_documenti(project, query, top_k=5):
    client = connect_weaviate()
    embeddings = create_embedding()
    store = WeaviateVectorStore(
        client=client,
        index_name=project,
        text_key="text",
        embedding=embeddings,
    )
    docs = store.similarity_search(query)
    client.close()
    return docs