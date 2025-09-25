import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from dashboard.service.ai import (LLM_CONFIGS, connect_weaviate,
                                  create_embedding, create_llm, embedding_docs,
                                  ingest_text, load_documents_from_dir, query_rag)
from dashboard.service.svn import svn_checkout

load_dotenv()

PREFIX = "RAG_VLLM_"


# --- Cluster Analysis Functions ---
def get_document_embeddings(client, project_name, limit=1000):
    """Retrieve document embeddings from Weaviate"""
    try:
        collection = client.collections.get(project_name)
        response = collection.query.fetch_objects(
            # limit=limit,
            include_vector=True
        )
        embeddings = []
        documents = []
        metadata = []
        
        for obj in response.objects:
            if obj.vector is not None:
                vec = obj.vector
                # print(vec)
                print(type[vec])
                # if isinstance(vec, dict):  # unwrap if needed
                #     vec = vec.get("vector", [])
                embeddings.append(vec)
                
                documents.append(obj.properties.get('text', ''))
                metadata.append({
                    'uuid': str(obj.uuid),
                    'source': obj.properties.get('source', ''),
                    'chunk_id': obj.properties.get('chunk_id', '')
                })
        
        return np.array(embeddings), documents, metadata
    except Exception as e:
        st.error(f"Error retrieving embeddings: {e}")
        return None, None, None


def perform_clustering(embeddings, method='kmeans', n_clusters=5, **kwargs):
    """Perform clustering on embeddings"""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    print(clusterer)
    print(type(embeddings[0]))
    print(embeddings[0])

    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels, clusterer


def reduce_dimensions(embeddings, method='tsne', n_components=2):
    """Reduce dimensionality for visualization"""
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def analyze_clusters(embeddings, documents, metadata, cluster_labels):
    """Analyze cluster characteristics"""
    unique_labels = np.unique(cluster_labels)
    cluster_info = []
    
    for label in unique_labels:
        if label == -1:  # Noise points in DBSCAN
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {label}"
        
        mask = cluster_labels == label
        cluster_docs = [documents[i] for i in range(len(documents)) if mask[i]]
        cluster_size = np.sum(mask)
        
        # Calculate cluster center (centroid)
        if cluster_size > 0:
            cluster_center = np.mean(embeddings[mask], axis=0)
            
            # Find most representative documents (closest to centroid)
            distances = []
            for i, embedding in enumerate(embeddings[mask]):
                dist = np.linalg.norm(embedding - cluster_center)
                distances.append(dist)
            
            # Get top 3 most representative documents
            top_indices = np.argsort(distances)[:3]
            representative_docs = [cluster_docs[i][:200] + "..." for i in top_indices]
            
            cluster_info.append({
                'label': label,
                'name': cluster_name,
                'size': cluster_size,
                'percentage': (cluster_size / len(documents)) * 100,
                'representative_docs': representative_docs
            })
    
    return cluster_info


def create_cluster_visualization(reduced_embeddings, cluster_labels, cluster_info):
    """Create interactive cluster visualization"""
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': cluster_labels,
        'cluster_name': [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
    })
    
    fig = px.scatter(
        df, x='x', y='y', 
        color='cluster_name',
        title='Document Clusters Visualization',
        hover_data=['cluster'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig


def cluster_based_retrieval(client, project_name, query, cluster_labels, embeddings, documents, metadata, top_k=5):
    """Enhanced retrieval using cluster information"""
    # Get query embedding
    # This is a placeholder - you'd need to implement query embedding using your embedding model
    # query_embedding = create_embedding(query)
    
    # For now, we'll use standard retrieval and then analyze which clusters are most relevant
    try:
        collection = client.collections.get(project_name)
        
        # Standard vector search
        response = collection.query.near_text(
            query=query,
            limit=top_k * 2  # Get more results to analyze cluster distribution
        )
        
        results = []
        cluster_distribution = {}
        
        for obj in response.objects:
            doc_uuid = str(obj.uuid)
            
            # Find which cluster this document belongs to
            doc_cluster = -1
            for i, meta in enumerate(metadata):
                if meta['uuid'] == doc_uuid:
                    doc_cluster = cluster_labels[i]
                    break
            
            cluster_distribution[doc_cluster] = cluster_distribution.get(doc_cluster, 0) + 1
            
            results.append({
                'content': obj.properties.get('text', ''),
                'source': obj.properties.get('source', ''),
                'cluster': doc_cluster,
                'score': getattr(obj, 'distance', 0)  # If available
            })
        
        return results[:top_k], cluster_distribution
    
    except Exception as e:
        st.error(f"Error in cluster-based retrieval: {e}")
        return [], {}


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
st.set_page_config(page_title="RAG Multi-Progetto con Cluster Analysis", layout="wide")
st.title("📚 RAG Multi-Progetto con Weaviate v4 + Cluster Analysis")

selected_name = st.selectbox("Choose a model:", [c["name"] for c in LLM_CONFIGS])

# Get config and build LLM
selected_config = next(c for c in LLM_CONFIGS if c["name"] == selected_name)

def clusterAnalysis():
    st.header("🔍 Cluster Analysis")
    
    projects = get_existing_projects()
    project_cluster = st.selectbox("Seleziona progetto per analisi:", projects, key="cluster_project")
    n_clusters =5 
    min_samples =5
    eps=0.1
    
    if project_cluster:
        col1, col2 = st.columns(2)
        
        with col1:
            clustering_method = st.selectbox(
                "Metodo di clustering:",
                ["kmeans", "dbscan", "agglomerative"]
            )
            
            if clustering_method in ["kmeans", "agglomerative"]:
                n_clusters = st.slider("Numero di cluster:", 2, 20, 5)
            else:
                eps = st.slider("DBSCAN eps:", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN min_samples:", 2, 20, 5)
        
        with col2:
            viz_method = st.selectbox(
                "Metodo di riduzione dimensionalità:",
                ["tsne", "pca"]
            )
            
            max_docs = st.number_input(
                "Massimo numero di documenti da analizzare:",
                min_value=50, max_value=5000, value=1000, step=50
            )
        
        if st.button("Esegui Cluster Analysis"):
            st.write(f"n_clustersg...{n_clusters}")
            
            with st.spinner("Caricamento embeddings..."):
                client = connect_weaviate()
                embeddings, documents, metadata = get_document_embeddings(
                    client, project_cluster, limit=max_docs
                )
                client.close()
            
             
            if embeddings is not None and len(embeddings) > 0:
                st.write(f"len(embeddings):{len(embeddings)}")
                
                with st.spinner("Esecuzione clustering...{n_clusters}"):
                    # Perform clustering
                    if clustering_method == "dbscan":
                        cluster_labels, clusterer = perform_clustering(
                            embeddings, method=clustering_method,
                            eps=eps, min_samples=min_samples
                        )
                    else:
                        cluster_labels, clusterer = perform_clustering(
                            embeddings, method=clustering_method,
                            n_clusters=n_clusters
                        )
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    sil_score = silhouette_score(embeddings, cluster_labels)
                    st.metric("Silhouette Score", f"{sil_score:.3f}")
                
                # Analyze clusters
                cluster_info = analyze_clusters(embeddings, documents, metadata, cluster_labels)
                
                # Display cluster information
                st.subheader("📊 Informazioni Cluster")
                
                cluster_df = pd.DataFrame([
                    {
                        'Cluster': info['name'],
                        'Dimensione': info['size'],
                        'Percentuale': f"{info['percentage']:.1f}%"
                    }
                    for info in cluster_info
                ])
                
                st.dataframe(cluster_df)
                
                # Visualize clusters
                with st.spinner("Creazione visualizzazione..."):
                    reduced_embeddings = reduce_dimensions(embeddings, method=viz_method)
                    fig = create_cluster_visualization(reduced_embeddings, cluster_labels, cluster_info)
                    st.plotly_chart(fig)
                
                # Show representative documents for each cluster
                st.subheader("📄 Documenti Rappresentativi per Cluster")
                
                for info in cluster_info:
                    if info['label'] != -1:  # Skip noise cluster for detailed view
                        with st.expander(f"{info['name']} ({info['size']} documenti)"):
                            for i, doc in enumerate(info['representative_docs'], 1):
                                st.write(f"**Doc {i}:** {doc}")
                
                # Store cluster information in session state for use in queries
                st.session_state['cluster_labels'] = cluster_labels
                st.session_state['cluster_embeddings'] = embeddings
                st.session_state['cluster_documents'] = documents
                st.session_state['cluster_metadata'] = metadata
                st.session_state['cluster_project'] = project_cluster
                
                st.success("✅ Cluster analysis completata!")
            
            else:
                st.error("Nessun embedding trovato per questo progetto.")

clusterAnalysis() 

# Add sidebar with cluster analysis info
if 'cluster_labels' in st.session_state:
    with st.sidebar:
        st.header("🔍 Cluster Info")
        st.write(f"**Progetto:** {st.session_state.get('cluster_project', 'N/A')}")
        
        unique_clusters = len(np.unique(st.session_state['cluster_labels']))
        st.write(f"**Clusters:** {unique_clusters}")
        
        total_docs = len(st.session_state['cluster_labels'])
        st.write(f"**Documenti:** {total_docs}")
        
        if st.button("🗑️ Pulisci Cluster Cache"):
            for key in ['cluster_labels', 'cluster_embeddings', 'cluster_documents', 'cluster_metadata', 'cluster_project']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Cache cluster pulita!")