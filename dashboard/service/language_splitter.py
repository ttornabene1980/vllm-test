import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Supported languages mapping
from langchain.text_splitter import Language
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


LANGUAGE_MAP = {
    "Python": Language.PYTHON,
    "Python 3": Language.PYTHON,
    "Java": Language.JAVA,
    "JavaScript": Language.JS,
    "TypeScript": Language.TS,
    "HTML": Language.HTML,
    "Markdown": Language.MARKDOWN,
}

EXTENSION_LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
}


def detect_language_from_extension(filename: str) -> Language | None:
    import os

    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in EXTENSION_LANGUAGE_MAP.keys():
        return EXTENSION_LANGUAGE_MAP[ext]
    else:
        return None

def detect_language(code: str) -> Language | None:
    lexer = guess_lexer(code)
    language_name = lexer.name
    # print(f"language_name:{language_name} {LANGUAGE_MAP}")
    if language_name in LANGUAGE_MAP.keys():
        return LANGUAGE_MAP[language_name]
    else:
        return None

def split_code_str(
    code: str, path:str, chunk_size=2000, chunk_overlap=100
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    chunks = splitter.split_text(code)
    metadata_template = {
        "filename": path
    }
    docs = [
        Document(page_content=chunk, metadata={**metadata_template, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    return docs

def split_code_with_metadata(
    path: str, chunk_size=2000, chunk_overlap=100
) -> List[Document]:
    filename = os.path.basename(path)

    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    # language_detect = detect_language(code)
    # print(f"language_detect:{language_detect}")
    language_extension = detect_language_from_extension(filename=path)
    # print(f"language_extension:{language_extension}")
    language_text = ""
    if language_extension != None:
        language_text = language_extension.upper()
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    # print(f"language_text:{language_text}")
    chunks = splitter.split_text(code)
    metadata_template = {
        "language": language_text,
        "filename": filename,
        "filepath": path,
    }
    docs = [
        Document(page_content=chunk, metadata={**metadata_template, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    return docs


# # 5️⃣ Esempio di utilizzo
# if __name__ == "__main__":
#     files = [
#         "/Users/tindarotornabene/develop/sorgente/vllm-test/volume_data/svn/ENTTBL/MATERIAL/BE/trunk/material-pnd-client/src/main/java/com/pnd/client/ApiClientV26.java",
#         "/Users/tindarotornabene/develop/sorgente/vllm-test/volume_data/svn/ENTTBL/MATERIAL/FE/trunk/material-frontend/src/main/webapp/scripts/app/entities/templateManager/templateManager-detail.controller.js",
#     ]

#     for file in files:
#         docs = split_code_with_metadata(file, chunk_size=2000, chunk_overlap=100)
#         # Stampare i chunk
#         for chunk in docs:
#             print(f"Chunk :\n{chunk}\n{'-'*40}")
#         # for f in files:
#         #     print(f"\nProcessing file: {f}\n{'='*60}")
#         # chunks = split_code_file(f, chunk_size=150, chunk_overlap=20)
#         # for i, c in enumerate(chunks):
#         #     print(f"Chunk {i+1}:\n{c}\n{'-'*40}")
