import os
import weaviate
from langchain_openai import ChatOpenAI 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.client import  WeaviateClient
from langchain_weaviate import WeaviateVectorStore
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

LLM_CONFIGS = [
    {
        "name": "DeepSeek Coder 6.7B",
        "model": "deepseek-coder-6.7b-instruct",
        "base_url": "http://192.168.1.98:8000/v1",
        "temperature": 0,
        "verbose": True,
    },
    {
        "name": "DeepSeek R1 Qwen3-8B",
        "model": "DeepSeek-R1-0528-Qwen3-8B",
        "base_url": "http://10.199.145.180:8080/v1",
        "temperature": 0,
        "verbose": True,
    },
    {
        "name": "OpenAI GPT-4o-mini",
        "model": "gpt-4o-mini",
        "base_url": None,
        "temperature": 0,
        "verbose": False,
    },
    {
        "name": "OpenAI GPT-4",
        "model": "gpt-4",
        "base_url": None,
        "temperature": 0,
        "verbose": False,
    },
]

EMBEDDING_CONFIGS = [
    {"name": "OpenAI Embeddings", "type": "openai", "model": "text-embedding-3-large"},
    {
        "name": "HuggingFace MiniLM",
        "type": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    {"name": "GPT4All Embeddings", "type": "gpt4all", "model": "ggml-model-q4_0.bin"},
]

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
    
def create_llm(config):
    print(config)
    kwargs = {
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "verbose": config.get("verbose", False),
    }
    if config.get("base_url"):
        kwargs["base_url"] = config["base_url"]

    return ChatOpenAI(**kwargs)

 
def embedding_docs(client:WeaviateClient, docs, project:str ):
   
    project2 = project.replace("/","_").replace(":","_").replace(".","_")
    print(f"Project:{project} -> {project2}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # adjust as needed (500–2000 is typical)
        chunk_overlap=100
    )
     
    loaded_docs = []
    for path in docs:
        print(f"Loading {path}... {os.path.isfile(path)}")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                print(f"Loaded {len(text)} characters from {path}")
                doc = Document(page_content=text, metadata={"source": path})
                chunks = splitter.split_documents([doc])
                loaded_docs.extend(chunks)
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


def create_embedding(config=None )->HuggingFaceEmbeddings:
    # if config["type"] == "openai":
    #     return OpenAIEmbeddings(model=config["model"])
    # elif config["type"] == "huggingface":
        return    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # elif config["type"] == "gpt4all":
    #     return GPT4AllEmbeddings(model_path=config["model"])
    # else:
    #     raise ValueError(f"Unknown embedding type: {config['type']}")


def llm_create():
    llm = ChatOpenAI(
        model="deepseek-coder-6.7b-instruct",
        base_url="http://192.168.1.98:8000/v1",
        verbose=True,
        temperature=0,
    )
    return llm


def llm_create_openai():
    return ChatOpenAI(model="gpt-4", temperature=0)

INCLUDED_EXTS = {".java", ".html", ".js", ".xml", ".yml", ".properties"}
EXCLUDED_DIRS = {".svn", ".git", "__pycache__"}

def load_documents_from_dir(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for file in filenames:
            _, ext = os.path.splitext(file)
            if ext.lower() in INCLUDED_EXTS:
                docs.append(os.path.join(dirpath, file))
    print(f"Loading docs:{len(docs)}...")
    return docs




from langchain.callbacks.base import BaseCallbackHandler


# 1️⃣ Capture reasoning in a variable
class ReasoningLogger(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def getLogs(self):
        return self.logs

    def on_agent_action(self, action, **kwargs):
        # Capture the Thought/Action/Action Input
        d = {"tool": action.tool, "tool_input": action.tool_input, "log": action.log}
        # print("Agent action:", d)
        self.logs.append(d)

    def on_agent_finish(self, finish, **kwargs):
        self.logs.append({"final_answer": finish.return_values})
