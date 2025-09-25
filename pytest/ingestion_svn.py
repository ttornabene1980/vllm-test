import operator
import os
from typing import List, Literal, TypedDict

import pandas as pd
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send

from dashboard.service.ai import (LLM_CONFIGS, create_llm,
                                  load_documents_from_dir, summarize_refine_chain)
from dashboard.service.language_splitter import split_code_with_metadata
from dashboard.service.vectordb import (connect_weaviate, embedding_docs,
                                        load_documents)

selected_config =  LLM_CONFIGS[0]

dest_path = "/Users/tindarotornabene/develop/sorgente/vllm-test/volume_data/svn/ENTTBL/MATERIAL/BE/trunk"
project = f"summary-{selected_config['model']}"

files = load_documents_from_dir(dest_path)

chunk_size = selected_config["max_tokens"] -100
documents = load_documents(files=files, project=dest_path, chunk_size=chunk_size )

# for doc in documents:
# print(f"Chunk:{doc.metadata}\n{doc.page_content} \n{'-'*80} \n{'-'*80} \n{'-'*80}\n")
# {doc.page_content}

print(f"documents chunk_size:{chunk_size} numerOfChunk:{len(documents)}")


llm = create_llm(selected_config )
print(f"summarize_refine_chain n:{len(documents)}")


summarize_refine_chain(project,llm, documents)
