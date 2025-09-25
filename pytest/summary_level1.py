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
                                  load_documents_from_dir, refine_chain, summarize_refine_chain)
from dashboard.service.language_splitter import split_code_with_metadata
from dashboard.service.vectordb import (connect_weaviate, embedding_docs,
                                        load_documents, load_pandas)

selected_config =  LLM_CONFIGS[0]

project = f"summary-level1-{selected_config['model']}"

chunk_size = selected_config["max_tokens"] -100
summary_repo_df = pd.read_json("summary.json")

print(summary_repo_df.head())
 
df_grouped = summary_repo_df.groupby('filename')['summary'].agg(' '.join).reset_index()

print(df_grouped)
df_grouped.to_html("summary_level1.html")



import os
from collections import defaultdict
import json

# Sample DataFrame
# df = pd.DataFrame({'summary': [...], 'filename': [...]})

# Step 1: Build the tree
def tree():
    return defaultdict(tree)

root = tree()
for _, row in df_grouped.iterrows():
    path_parts = os.path.normpath(row['filename']).split(os.sep)
    current = root
    for part in path_parts[:-1]:  # directories only
        current = current[part]
    if 'summaries' not in current:
        current['summaries'] = []
    current['summaries'].append(row['summary'])

# Step 2: Aggregate summaries and prune leaf summaries
def aggregate_and_prune_with_id(node, counter):
    aggregated = []
    for key, child in node.items():
        if isinstance(child, dict):
            child_agg, counter = aggregate_and_prune_with_id(child, counter)
            aggregated.extend(child_agg)
            if 'summaries' in child:
                del child['summaries']
    if 'summaries' in node:
        aggregated.extend(node['summaries'])
        del node['summaries']
    node['aggregated_summaries'] = aggregated
    node['id'] = counter
    counter += 1
    return aggregated, counter

aggregate_and_prune_with_id(root, counter=1)

# Step 3: Convert defaultdict to dict
def dictify(d):
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    return d

tree_dict = dictify(root)

# Step 4: Optional - flatten for pandas HTML
def flatten_tree_with_id(node, path=None):
    if path is None:
        path = []
    rows = []
    for key, child in node.items():
        if isinstance(child, dict):
            rows.append({
                'id': child['id'],
                'path': '/'.join(path + [key]),
                'aggregated_summaries': ' '.join(child['aggregated_summaries'])
            })
            rows.extend(flatten_tree_with_id(child, path + [key]))
    return rows

flat_data = flatten_tree_with_id(tree_dict)
df_flat = pd.DataFrame(flat_data)

# Save to HTML
df_flat.to_html('aggregated_summaries_with_id.html', index=False)
df_flat.to_json('aggregated_summaries_with_id.json', index=False)

print( df_flat.head() )
documents = load_pandas(df_flat, project,chunk_size= chunk_size)
print(f"documents chunk_size:{chunk_size} numerOfChunk:{len(documents)}")

llm = create_llm(selected_config )
print(f"refine_chain n:{len(documents)}")
refine_chain(project,llm, documents,1)
