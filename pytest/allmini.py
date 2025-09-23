# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(["Hello world"])
# print(embeddings.shape)

from dashboard.service.ai import connect_weaviate, embedding_docs, load_documents_from_dir
new_project = "volume_data/svn/ENTTBL/MATERIAL/BE/trunk"

docs = load_documents_from_dir( new_project  )
client = connect_weaviate()
embedding_docs(client,docs = docs ,  project=new_project )
client.close()
        
        