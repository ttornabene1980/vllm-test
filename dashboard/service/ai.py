import os

import pandas as pd
import weaviate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from transformers import pipeline
from weaviate.client import WeaviateClient
import re
from dashboard.service.vectordb import connect_weaviate, create_embedding

LLM_CONFIGS = [
    {
        "name": "DeepSeek R1 Qwen3-8B",
        "model": "DeepSeek-R1-0528-Qwen3-8B",
        "base_url": "http://10.199.145.180:8080/v1",
        "temperature": 0,
        "verbose": True,
        "max_tokens":32768
    },
    {
        "name": "DeepSeek Coder 6.7B",
        "model": "deepseek-coder-6.7b-instruct",
        "base_url": "http://192.168.1.98:8000/v1",
        "temperature": 0,
        "verbose": True,
         "max_tokens":8192
    },
    {
        "name": "OpenAI GPT-4o-mini",
        "model": "gpt-4o-mini",
        "base_url": None,
        "temperature": 0,
        "verbose": False,
         "max_tokens":128000
    },
    {
        "name": "OpenAI GPT-4",
        "model": "gpt-4",
        "base_url": None,
        "temperature": 0,
        "verbose": False,
         "max_tokens":32000
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


def create_llm(config ,max_tokens=-1):
    print(f"create_llm:{config}" )
    kwargs = {
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "verbose": config.get("verbose", False),
        # "max_tokens": max_token
    }
    if config.get("base_url"):
        kwargs["base_url"] = config["base_url"]

    kwargs["base_url"] = config["base_url"]
    if( max_tokens > 0 ):
        kwargs["max_tokens"] = max_tokens
        
    # kwargs["verbose"] = True
    
    return ChatOpenAI(**kwargs)



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


INCLUDED_EXTS = {".java", ".html", ".js" } 
# , ".xml", ".yml", ".properties"}
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



from enum import Enum


def query_rag(query: str, project: str , selected_config , typeQA:str="stuff" ,  top_k: int = 10 ):
    client = connect_weaviate()
    embeddings = create_embedding()
    
   
    
    store = WeaviateVectorStore(
        client=client,
        index_name=project,
        text_key="text",
        embedding=embeddings,
    )
    response = None
    
    if( typeQA == 'map_reduce'):
        retriever = store.as_retriever(search_kwargs={"k": top_k})
        docs = store._collection
        
        llm = create_llm(selected_config,150)
        global_know = []
        i = 0
        for chunk in docs:
            response_chunk = llm.invoke(f"Summarize this chunk in max 100 tokens {query}:\n{chunk.page_content}")
            response_chunk = llm.invoke(f"Classifica this chunk in domain(spiegazione del domain),service,restcontroller,api,test  {query}:\n{chunk.page_content}")
            print(response_chunk)
            global_know.append(response_chunk)
        #    second round tenendo conto del precedente metti
        # riformula il summari tenendo conto della precedente risposta = ultimo chunk
            response_chunk = llm.invoke(f"Summarize this chunk in max 100 tokens riformula il summari tenendo conto della precedente risposta:\n{chunk.page_content}")
    
        # final_prompt = "Combine the summaries into a concise answer:\n" + "\n".join(answers)
        # response = llm.invoke(final_prompt)
    elif(typeQA =='stuff'):
        llm = create_llm(selected_config)
        # Combine retrieved texts
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
        You are an expert assistant. Use the following context to answer the question concisely.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        response = llm.invoke(prompt)
       
    print(response)
    print(f"✅ Retrieved response '{response}'")
    client.close()
    return response 
    


def summarize_refine_chain(project:str, llm, chunks: list[Document]):
    # initial summarization prompt
    # summarize_prompt = ChatPromptTemplate.from_template(
    #     "Summarize the following text in under 50 words:\n\n{context} "
    # )
    
    summarize_prompt = ChatPromptTemplate.from_template(
    """Summarize the following  code in one concise paragraph max 100 words.
    Do not use `<think>` tags, do not include any internal reasoning:
    
    Code:
    {context}"""
    )
        
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    summaries = []
    chunk = chunks[0]
    
    # Step 1: summarize first chunk
    summary = summarize_chain.invoke(
        {
            "context": chunk.page_content,
            "filename": chunk.metadata.get("filepath"),
        }
    )
    
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    summaryRatio = f"{len(summary)}/{len(chunk.page_content)}"
    summary_part = {  "summaryRatio":summaryRatio ,  "summary":summary  ,"context": chunk.page_content,  "language": chunk.metadata.get("language"),  "filename": chunk.metadata.get("filepath")  }
    summaries.append(summary_part)
    
    
    dataseet = pd.DataFrame(summaries)
    dataseet.to_html(f"{project}.html")
    dataseet.to_json(f"{project}.json")
    
    # Step 2: for each next chunk, refine
    i=1
    for chunk in chunks[1:]:
        summary = summarize_chain.invoke(
            {
                "context": chunk.page_content,
                "filename": chunk.metadata.get("filepath"),
            }
        )
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        summaryRatio = f"{len(summary)}/{len(chunk.page_content)}"
        summary_part = {  "summaryRatio":summaryRatio , "summary":summary  ,"context": chunk.page_content,  "language": chunk.metadata.get("language"),  "filename": chunk.metadata.get("filepath")  }
        summaries.append(summary_part)
        i= i+1
        
        if i%10 == 0:
            dataseet = pd.DataFrame(summaries)
            dataseet.to_html(f"{project}.html")
            dataseet.to_json(f"{project}.json")
        print(f"{i})/{len(chunks)}-")
        
    dataseet = pd.DataFrame(summaries)
    dataseet.to_html(f"{project}.html")
    dataseet.to_json(f"{project}.json")
            
    return summaries





def refine_chain(project:str,  llm, chunks: list[Document],level:int ):
    summaryFile = f"{project}-level{level}.html"
    refine_prompt = ChatPromptTemplate.from_template(
    """Refine the following 
    {context} 
    in one concise refine summary .
    Do not use `<think>` tags, do not include any internal reasoning:
    """
    )
    refine_chain = refine_prompt | llm | StrOutputParser()
    summaries = []
    chunk = chunks[0]
    summary = refine_chain.invoke(
        {
            "context": chunk.page_content,
            "filename": chunk.metadata.get("filename"),
        }
    )
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    summaryRatio = f"{len(summary)}/{len(chunk.page_content)}"
    summary_part = {  "summaryRatio":summaryRatio ,  "summary":summary  ,"context": chunk.page_content,  "language": chunk.metadata.get("language"),  "filename": chunk.metadata.get("filename")  }
    summaries.append(summary_part)
    dataseet = pd.DataFrame(summaries)
    dataseet.to_html(f"{summaryFile}.html")
    dataseet.to_json(f"{summaryFile}.json")
    
    i=1
    for chunk in chunks[1:]:
        summary = refine_chain.invoke(
            {
                "context": chunk.page_content,
                "filename": chunk.metadata.get("filename"),
            }
        )
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        summaryRatio = f"{len(summary)}/{len(chunk.page_content)}"
        summary_part = {  "summaryRatio":summaryRatio , "summary":summary  ,"context": chunk.page_content,  "language": chunk.metadata.get("language"),  "filename": chunk.metadata.get("filename")  }
        summaries.append(summary_part)
        i= i+1
        if i%10 == 0:
            dataseet = pd.DataFrame(summaries)
            dataseet.to_html(f"{summaryFile}.html")
            dataseet.to_json(f"{summaryFile}.json")
        print(f"{i})/{len(chunks)}-")
        
    dataseet = pd.DataFrame(summaries)
    dataseet.to_html(f"{summaryFile}.html")
    dataseet.to_json(f"{summaryFile}.json")
            
    return summaries




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

