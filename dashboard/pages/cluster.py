import streamlit as st
import numpy as np
import pandas as pd
import json
import pickle
from typing import List, Dict, Any
import io
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Configure page
st.set_page_config(
    page_title="Embedding Map-Reduce Processor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #28a745;
}
.warning-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

# Mock LLM for demonstration (replace with actual LLM in production)
class MockLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        
    def __call__(self, prompt: str) -> str:
        # Simulate processing time
        time.sleep(0.5)
        
        # Analyze prompt type and generate appropriate response
        if "SUMMARY" in prompt.upper() or "combine" in prompt.lower():
            return self._generate_summary_response(prompt)
        else:
            return self._generate_map_response(prompt)
    
    def _generate_map_response(self, prompt: str) -> str:
        # Extract some info from the prompt
        lines = prompt.split('\n')
        doc_count = len([l for l in lines if 'Document:' in l])
        similarity_scores = [l for l in lines if 'Similarity:' in l]
        
        patterns = [
            f"Identified {doc_count} documents in this chunk",
            f"Found {np.random.randint(2, 8)} distinct clustering patterns",
            f"Average similarity score: {np.random.uniform(0.3, 0.9):.3f}",
            f"Detected {np.random.randint(1, 4)} high-similarity clusters (>0.8)",
            f"Vector dimensionality analysis shows {np.random.randint(300, 800)} effective dimensions"
        ]
        
        return "CHUNK ANALYSIS:\n" + "\n".join(np.random.choice(patterns, 3, replace=False))
    
    def _generate_summary_response(self, prompt: str) -> str:
        summaries = [
            """FINAL SYNTHESIS:
            
Overall Patterns:
- Discovered 3 major document clusters with high internal similarity (>0.85)
- Identified 15-20 outlier documents with unique vector patterns
- Found strong semantic relationships in 60% of the dataset
- Detected hierarchical clustering structure with 2-3 main branches

Key Insights:
- Cluster 1: Technical documents (similarity: 0.92)
- Cluster 2: Marketing content (similarity: 0.88) 
- Cluster 3: Mixed general content (similarity: 0.76)

Recommendations:
- Focus on Cluster 1 for technical similarity searches
- Consider separate processing for outlier documents
- Implement hierarchical search for better relevance""",

            """COMPREHENSIVE ANALYSIS SUMMARY:

Pattern Discovery:
- Identified 4 distinct semantic clusters across the embedding space
- Found 85% of documents fall into clear similarity groups
- Discovered potential duplicate/near-duplicate content (similarity >0.95)

Statistical Insights:
- Mean cosine similarity: 0.74
- Standard deviation: 0.18
- Maximum cluster density: 0.91
- Minimum inter-cluster distance: 0.23

Strategic Recommendations:
- Implement cluster-based indexing for faster retrieval
- Consider dimensionality reduction to 512 dimensions
- Apply threshold-based filtering at 0.70 similarity""",

            """EMBEDDING ANALYSIS RESULTS:

Structural Findings:
- Clear separation between domain-specific content types
- Strong intra-cluster coherence (avg similarity: 0.87)
- Moderate inter-cluster relationships (avg: 0.45)

Quality Metrics:
- High-quality embeddings detected: 78%
- Potential noise/outliers: 12%
- Borderline cases requiring review: 10%

Action Items:
- Refine similarity thresholds for each cluster
- Investigate outlier documents for data quality
- Consider ensemble approaches for borderline cases"""
        ]
        
        return np.random.choice(summaries)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None

# Main title
st.markdown('<h1 class="main-header">🔍 Embedding Vector Map-Reduce Processor</h1>', unsafe_allow_html=True)
st.markdown("**Process large embedding datasets using map-reduce techniques to overcome token limits**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Processing parameters
    chunk_size = st.slider("Chunk Size (characters)", 500, 8000, 2000, 
                          help="Size of each text chunk for processing")
    chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 200,
                             help="Overlap between chunks to maintain context")
    max_tokens_per_chunk = st.slider("Max Tokens per Chunk", 1000, 16000, 4000,
                                   help="Maximum tokens for LLM processing")
    
    st.divider()
    
    # LLM Configuration
    st.subheader("🤖 LLM Settings")
    llm_provider = st.selectbox("LLM Provider", 
                               ["Mock (Demo)", "OpenAI", "Anthropic", "Local Model"],
                               help="Select your language model provider")
    
    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
    elif llm_provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password")
        model_name = st.selectbox("Model", ["claude-3-sonnet", "claude-3-opus"])
    
    st.divider()
    
    # Analysis options
    st.subheader("📊 Analysis Options")
    include_visualization = st.checkbox("Include Visualizations", True)
    include_clustering = st.checkbox("Perform Clustering", True)
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📤 Data Input", "🔄 Processing", "📊 Results", "💾 Export"])

with tab1:
    st.header("Data Input Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Data")
        
        upload_option = st.radio("File Format", 
                                ["JSON", "Pickle", "CSV", "NumPy"])
        
        uploaded_file = st.file_uploader(
            f"Upload {upload_option} file",
            type=['json', 'pkl', 'csv', 'npy'],
            help=f"Upload your embedding data in {upload_option} format"
        )
        
        if uploaded_file:
            try:
                if upload_option == "JSON":
                    data = json.load(uploaded_file)
                elif upload_option == "Pickle":
                    data = pickle.load(uploaded_file)
                elif upload_option == "CSV":
                    df = pd.read_csv(uploaded_file)
                    data = {
                        "vectors": df.select_dtypes(include=[np.number]).values.tolist(),
                        "metadata": df.iloc[:, 0].astype(str).tolist(),
                        "similarities": np.random.random(len(df)).tolist()
                    }
                elif upload_option == "NumPy":
                    vectors = np.load(uploaded_file)
                    data = {
                        "vectors": vectors.tolist(),
                        "metadata": [f"doc_{i}" for i in range(len(vectors))],
                        "similarities": np.random.random(len(vectors)).tolist()
                    }
                
                st.session_state.embeddings_data = data
                st.success(f"✅ Loaded {len(data.get('vectors', []))} embedding vectors")
                
                # Display data info
                if 'vectors' in data:
                    vector_dim = len(data['vectors'][0]) if data['vectors'] else 0
                    st.info(f"Vector dimension: {vector_dim}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.subheader("🎲 Generate Sample Data")
        
        # Sample data parameters
        n_samples = st.number_input("Number of samples", 10, 10000, 500)
        vector_dim = st.number_input("Vector dimension", 50, 2048, 384)
        n_clusters = st.number_input("Number of clusters", 2, 10, 3)
        
        if st.button("🚀 Generate Sample Dataset", type="primary"):
            with st.spinner("Generating sample embeddings..."):
                # Create clustered sample data
                cluster_centers = np.random.random((n_clusters, vector_dim))
                vectors = []
                metadata = []
                true_labels = []
                
                for i in range(n_samples):
                    cluster_id = np.random.randint(0, n_clusters)
                    # Add noise to cluster center
                    vector = cluster_centers[cluster_id] + np.random.normal(0, 0.1, vector_dim)
                    vectors.append(vector.tolist())
                    metadata.append(f"document_{i}_cluster_{cluster_id}")
                    true_labels.append(cluster_id)
                
                # Calculate similarities (distance to nearest cluster center)
                similarities = []
                for i, vector in enumerate(vectors):
                    distances = [np.linalg.norm(np.array(vector) - center) 
                               for center in cluster_centers]
                    similarity = 1 / (1 + min(distances))  # Convert distance to similarity
                    similarities.append(similarity)
                
                sample_data = {
                    "vectors": vectors,
                    "metadata": metadata,
                    "similarities": similarities,
                    "true_labels": true_labels
                }
                
                st.session_state.embeddings_data = sample_data
                st.success(f"✅ Generated {n_samples} synthetic embeddings with {n_clusters} clusters")
    
    # Display loaded data info
    if st.session_state.embeddings_data:
        st.subheader("📋 Dataset Overview")
        data = st.session_state.embeddings_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Vectors", len(data.get('vectors', [])))
        with col2:
            vector_dim = len(data['vectors'][0]) if data.get('vectors') else 0
            st.metric("📐 Vector Dimension", vector_dim)
        with col3:
            if 'similarities' in data:
                avg_sim = np.mean(data['similarities'])
                st.metric("📈 Avg Similarity", f"{avg_sim:.3f}")
        with col4:
            data_size_mb = len(str(data)) / (1024 * 1024)
            st.metric("💾 Data Size", f"{data_size_mb:.2f} MB")
        
        # Show sample data
        with st.expander("👀 Preview Data"):
            preview_df = pd.DataFrame({
                'Metadata': data.get('metadata', [])[:10],
                'Vector (first 5 dims)': [str(v[:5]) + '...' for v in data.get('vectors', [])[:10]],
                'Similarity': data.get('similarities', [])[:10]
            })
            st.dataframe(preview_df)

with tab2:
    st.header("🔄 Map-Reduce Processing")
    
    if not st.session_state.embeddings_data:
        st.warning("⚠️ Please load or generate data in the Data Input tab first.")
    else:
        data = st.session_state.embeddings_data
        
        # Processing setup
        st.subheader("Processing Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            processing_mode = st.radio("Processing Mode", 
                                     ["Full Map-Reduce", "Simple Chunking", "Hierarchical"])
        with col2:
            analysis_focus = st.multiselect("Analysis Focus", 
                                          ["Similarity Patterns", "Clustering", "Outlier Detection", 
                                           "Semantic Groups", "Quality Assessment"],
                                          default=["Similarity Patterns", "Clustering"])
        
        # Convert embeddings to processable text
        def embeddings_to_text(embeddings_data, include_full_vectors=False):
            text_chunks = []
            vectors = embeddings_data.get('vectors', [])
            metadata = embeddings_data.get('metadata', [])
            similarities = embeddings_data.get('similarities', [])
            
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                similarity = similarities[i] if i < len(similarities) else 0
                
                if include_full_vectors:
                    vector_str = ','.join([f"{x:.6f}" for x in vector])
                else:
                    # Use only first 10 dimensions to save space
                    vector_str = ','.join([f"{x:.4f}" for x in vector[:10]])
                
                chunk_text = f"""Document ID: {meta}
Vector Sample: [{vector_str}{'...' if not include_full_vectors else ''}]
Similarity Score: {similarity:.6f}
Vector Norm: {np.linalg.norm(vector):.6f}
---"""
                text_chunks.append(chunk_text)
            
            return '\n'.join(text_chunks)
        
        # Processing button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Convert to text
                status_text.text("Converting embeddings to text format...")
                progress_bar.progress(10)
                
                full_text = embeddings_to_text(data)
                
                # Step 2: Initialize text splitter
                status_text.text("Splitting text into chunks...")
                progress_bar.progress(20)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                
                chunks = text_splitter.split_text(full_text)
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                st.info(f"📦 Created {len(documents)} chunks for processing")
                progress_bar.progress(30)
                
                # Step 3: Set up prompts
                status_text.text("Setting up analysis prompts...")
                
                map_template = f"""
Analyze this chunk of embedding data focusing on: {', '.join(analysis_focus)}

Data chunk:
{{docs}}

Provide analysis covering:
- High similarity patterns (threshold: {similarity_threshold})
- Document groupings and clusters
- Notable vector characteristics
- Quality indicators
- Outliers or anomalies

Format your response as a structured analysis:
"""
                
                reduce_template = """
Synthesize these individual chunk analyses into a comprehensive report:

{doc_summaries}

Create a final summary including:
- Overall dataset patterns and structure
- Key similarity clusters identified
- Quality assessment and recommendations
- Statistical insights
- Actionable recommendations for usage

COMPREHENSIVE SUMMARY:
"""
                
                map_prompt = PromptTemplate.from_template(map_template)
                reduce_prompt = PromptTemplate.from_template(reduce_template)
                
                progress_bar.progress(40)
                
                # Step 4: Initialize LLM
                status_text.text("Initializing language model...")
                
                if llm_provider == "Mock (Demo)":
                    llm = MockLLM()
                elif llm_provider == "OpenAI" and 'api_key' in locals():
                    # In production, uncomment and configure:
                    # from langchain.llms import OpenAI
                    # llm = OpenAI(openai_api_key=api_key, model_name=model_name)
                    llm = MockLLM()  # Fallback to mock for demo
                else:
                    llm = MockLLM()  # Fallback to mock
                
                progress_bar.progress(50)
                
                # Step 5: Create processing chains
                status_text.text("Creating processing chains...")
                
                map_chain = LLMChain(llm=llm, prompt=map_prompt)
                reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
                
                combine_documents_chain = StuffDocumentsChain(
                    llm_chain=reduce_chain,
                    document_variable_name="doc_summaries"
                )
                
                reduce_documents_chain = ReduceDocumentsChain(
                    combine_documents_chain=combine_documents_chain,
                    collapse_documents_chain=combine_documents_chain,
                    token_max=max_tokens_per_chunk
                )
                
                map_reduce_chain = MapReduceDocumentsChain(
                    llm_chain=map_chain,
                    reduce_documents_chain=reduce_documents_chain,
                    document_variable_name="docs",
                    return_intermediate_steps=True
                )
                
                progress_bar.progress(60)
                
                # Step 6: Execute map-reduce
                status_text.text("Executing map-reduce analysis...")
                
                start_time = time.time()
                result = map_reduce_chain({"input_documents": documents})
                processing_time = time.time() - start_time
                
                progress_bar.progress(90)
                
                # Step 7: Prepare results
                status_text.text("Preparing results...")
                
                # Add metadata to results
                result['processing_metadata'] = {
                    'processing_time': processing_time,
                    'total_chunks': len(documents),
                    'total_vectors': len(data.get('vectors', [])),
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'analysis_focus': analysis_focus,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.processed_results = result
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success(f"🎉 Analysis completed in {processing_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.exception(e)

with tab3:
    st.header("📊 Analysis Results")
    
    if not st.session_state.processed_results:
        st.info("ℹ️ No results available. Please run the processing first.")
    else:
        result = st.session_state.processed_results
        metadata = result.get('processing_metadata', {})
        
        # Results summary
        st.subheader("📈 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
        with col2:
            st.metric("📦 Chunks Processed", metadata.get('total_chunks', 0))
        with col3:
            st.metric("📊 Total Vectors", metadata.get('total_vectors', 0))
        with col4:
            efficiency = metadata.get('total_vectors', 1) / max(metadata.get('processing_time', 1), 0.1)
            st.metric("🚀 Vectors/Second", f"{efficiency:.0f}")
        
        # Main results
        st.subheader("🎯 Final Analysis")
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write(result["output_text"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Intermediate steps
        if result.get("intermediate_steps"):
            with st.expander("🔍 Detailed Chunk Analysis", expanded=False):
                for i, step in enumerate(result["intermediate_steps"]):
                    st.markdown(f"**📝 Chunk {i+1} Analysis:**")
                    st.write(step)
                    if i < len(result["intermediate_steps"]) - 1:
                        st.divider()
        
        # Visualizations
        if include_visualization and st.session_state.embeddings_data:
            st.subheader("📊 Data Visualizations")
            
            data = st.session_state.embeddings_data
            vectors = np.array(data.get('vectors', []))
            
            if len(vectors) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # PCA visualization
                    if vectors.shape[1] > 2:
                        pca = PCA(n_components=2)
                        vectors_2d = pca.fit_transform(vectors)
                        
                        fig = px.scatter(
                            x=vectors_2d[:, 0], 
                            y=vectors_2d[:, 1],
                            color=data.get('similarities', [0] * len(vectors)),
                            title="PCA Visualization",
                            labels={'color': 'Similarity'},
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Similarity distribution
                    similarities = data.get('similarities', [])
                    if similarities:
                        fig = px.histogram(
                            x=similarities,
                            nbins=30,
                            title="Similarity Score Distribution",
                            labels={'x': 'Similarity Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Clustering visualization
                if include_clustering and len(vectors) > 10:
                    st.subheader("🎯 Clustering Analysis")
                    
                    n_clusters = min(5, len(vectors) // 10)  # Reasonable number of clusters
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(vectors)
                    
                    if vectors.shape[1] > 2:
                        pca = PCA(n_components=2)
                        vectors_2d = pca.fit_transform(vectors)
                        
                        fig = px.scatter(
                            x=vectors_2d[:, 0], 
                            y=vectors_2d[:, 1],
                            color=cluster_labels.astype(str),
                            title=f"K-Means Clustering (k={n_clusters})",
                            labels={'color': 'Cluster'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    cluster_stats = []
                    for i in range(n_clusters):
                        mask = cluster_labels == i
                        cluster_vectors = vectors[mask]
                        if len(cluster_vectors) > 1:
                            # Calculate intra-cluster similarity
                            sim_matrix = cosine_similarity(cluster_vectors)
                            avg_similarity = (sim_matrix.sum() - sim_matrix.trace()) / (len(cluster_vectors) * (len(cluster_vectors) - 1))
                        else:
                            avg_similarity = 1.0
                        
                        cluster_stats.append({
                            'Cluster': i,
                            'Size': mask.sum(),
                            'Avg Intra-Similarity': avg_similarity,
                            'Percentage': (mask.sum() / len(vectors)) * 100
                        })
                    
                    cluster_df = pd.DataFrame(cluster_stats)
                    st.dataframe(cluster_df, use_container_width=True)

with tab4:
    st.header("💾 Export Results")
    
    if not st.session_state.processed_results:
        st.info("ℹ️ No results to export. Please run the processing first.")
    else:
        result = st.session_state.processed_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Export Options")
            
            export_format = st.selectbox("Format", ["JSON", "Text", "CSV", "PDF Report"])
            include_intermediate = st.checkbox("Include intermediate steps", True)
            include_metadata = st.checkbox("Include processing metadata", True)
            
            # Prepare export data
            export_data = {
                "final_analysis": result["output_text"],
                "processing_timestamp": datetime.now().isoformat()
            }
            
            if include_intermediate and result.get("intermediate_steps"):
                export_data["intermediate_steps"] = result["intermediate_steps"]
            
            if include_metadata and result.get("processing_metadata"):
                export_data["processing_metadata"] = result["processing_metadata"]
            
            # Generate export file
            if export_format == "JSON":
                export_content = json.dumps(export_data, indent=2)
                filename = f"embedding_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                mime_type = "application/json"
            
            elif export_format == "Text":
                export_lines = [
                    "EMBEDDING MAP-REDUCE ANALYSIS RESULTS",
                    "=" * 50,
                    f"Generated: {export_data['processing_timestamp']}",
                    "",
                    "FINAL ANALYSIS:",
                    "-" * 20,
                    export_data["final_analysis"],
                ]
                
                if include_intermediate and "intermediate_steps" in export_data:
                    export_lines.extend([
                        "",
                        "INTERMEDIATE STEPS:",
                        "-" * 20,
                    ])
                    for i, step in enumerate(export_data["intermediate_steps"]):
                        export_lines.extend([f"Chunk {i+1}:", step, ""])
                
                export_content = "\n".join(export_lines)
                filename = f"embedding_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                mime_type = "text/plain"
            
            # Download button
            st.download_button(
                label=f"📥 Download {export_format}",
                data=export_content,
                file_name=filename,
                mime=mime_type,
                use_container_width=True
            )
        
        with col2:
            st.subheader("📊 Results Summary")
            
            # Create summary stats
            metadata = result.get('processing_metadata', {})
            
            summary_stats = {
                "Total Processing Time": f"{metadata.get('processing_time', 0):.2f} seconds",
                "Chunks Processed": metadata.get('total_chunks', 0),
                "Vectors Analyzed": metadata.get('total_vectors', 0),
                "Analysis Focus": ', '.join(metadata.get('analysis_focus', [])),
                "Chunk Size": f"{metadata.get('chunk_size', 0)} characters",
                "Processing Efficiency": f"{metadata.get('total_vectors', 1) / max(metadata.get('processing_time', 1), 0.1):.0f} vectors/sec"
            }
            
            for key, value in summary_stats.items():
                st.metric(key, value)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🔧 Production Setup Notes</h4>
    <p>
        • Replace MockLLM with actual language models (OpenAI, Anthropic, etc.)<br>
        • Configure proper API keys for production use<br>
        • Adjust chunk sizes based on your specific embedding dimensions<br>
        • Consider implementing caching for large datasets<br>
        • Add error handling and retry logic for robust production deployment
    </p>
    <p><strong>Built with Streamlit • LangChain • Plotly</strong></p>
</div>
""", unsafe_allow_html=True)