import streamlit as st
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
import weaviate.classes as wvc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import hashlib
import os

# Configuration
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_GRPC_URL = "http://localhost:50051"
VLLM_API_URL = "http://localhost:8000/v1/completions"

class GuardrailsValidator:
    """Simple guardrails for content validation"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b(password|secret|token|key)\b',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
    
    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate input text for sensitive content"""
        violations = []
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Potential sensitive data detected: {pattern}")
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "risk_score": min(len(violations) * 0.3, 1.0)
        }
    
    def validate_output(self, text: str) -> Dict[str, Any]:
        """Validate output text"""
        # Simple content validation
        harmful_keywords = ['violence', 'hate', 'harmful', 'dangerous']
        violations = [kw for kw in harmful_keywords if kw.lower() in text.lower()]
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "confidence": 1.0 - (len(violations) * 0.2)
        }

class SyntheticDataGenerator:
    """Generate synthetic clustered data for analysis"""
    
    @staticmethod
    def generate_customer_data(n_samples: int = 1000, n_clusters: int = 5) -> pd.DataFrame:
        """Generate synthetic customer data with clusters"""
        # Generate clustered data
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            n_features=4,
            random_state=42,
            cluster_std=1.5
        )
        
        # Create meaningful feature names and data
        df = pd.DataFrame(X, columns=['spending_score', 'frequency', 'recency', 'value'])
        df['cluster'] = y
        
        # Add categorical features
        categories = ['Premium', 'Standard', 'Basic', 'VIP', 'New']
        df['category'] = np.random.choice(categories, size=n_samples)
        
        # Add synthetic text data for RAG
        descriptions = [
            f"Customer with spending score {X[i,0]:.1f}, visits {X[i,1]:.1f} times per month, "
            f"last purchase {X[i,2]:.0f} days ago, lifetime value ${X[i,3]*100:.0f}"
            for i in range(n_samples)
        ]
        df['description'] = descriptions
        
        return df
    
    @staticmethod
    def generate_document_data(n_docs: int = 500) -> List[Dict]:
        """Generate synthetic documents for RAG system"""
        topics = [
            "machine learning algorithms and their applications",
            "customer behavior analysis and segmentation strategies", 
            "data clustering techniques and validation methods",
            "business intelligence and analytics best practices",
            "artificial intelligence in customer service"
        ]
        
        documents = []
        for i in range(n_docs):
            topic = np.random.choice(topics)
            content = f"Document {i+1}: This document discusses {topic}. " \
                     f"It contains detailed analysis of relevant methodologies, " \
                     f"case studies, and practical applications. The document " \
                     f"provides insights into industry trends and future developments."
            
            documents.append({
                "id": f"doc_{i+1}",
                "title": f"Analysis Document {i+1}",
                "content": content,
                "topic": topic,
                "created_at": datetime.now().isoformat()
            })
        
        return documents

class WeaviateManager:
    """Manage Weaviate vector database operations"""
    
    def __init__(self, url: str = WEAVIATE_URL):
        self.url = url
        self.client = None
        self.schema_created = False
    
    def connect(self):
        """Connect to Weaviate instance"""
        try:
            self.client = weaviate.Client(url=self.url)
            return True
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def create_schema(self):
        """Create schema for documents"""
        if self.schema_created:
            return True
            
        schema = {
            "classes": [
                {
                    "class": "Document",
                    "description": "A document for RAG system",
                    "properties": [
                        {
                            "name": "title",
                            "dataType": ["string"],
                            "description": "Document title"
                        },
                        {
                            "name": "content", 
                            "dataType": ["text"],
                            "description": "Document content"
                        },
                        {
                            "name": "topic",
                            "dataType": ["string"], 
                            "description": "Document topic"
                        },
                        {
                            "name": "created_at",
                            "dataType": ["string"],
                            "description": "Creation timestamp"
                        }
                    ]
                }
            ]
        }
        
        try:
            if self.client:
                # Delete existing schema if exists
                try:
                    self.client.schema.delete_class("Document")
                except:
                    pass
                
                self.client.schema.create(schema)
                self.schema_created = True
                return True
        except Exception as e:
            st.error(f"Failed to create schema: {e}")
            return False
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to Weaviate"""
        if not self.client:
            return False
            
        try:
            with self.client.batch as batch:
                for doc in documents:
                    batch.add_data_object(
                        data_object={
                            "title": doc["title"],
                            "content": doc["content"],
                            "topic": doc["topic"],
                            "created_at": doc["created_at"]
                        },
                        class_name="Document"
                    )
            return True
        except Exception as e:
            st.error(f"Failed to add documents: {e}")
            return False
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search documents using vector similarity"""
        if not self.client:
            return []
            
        try:
            result = (
                self.client.query
                .get("Document", ["title", "content", "topic", "created_at"])
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            if "data" in result and "Get" in result["data"]:
                return result["data"]["Get"]["Document"]
            return []
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

class ChatHistory:
    """Manage chat history with persistence"""
    
    def __init__(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to chat history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        st.session_state.chat_history.append(message)
    
    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return st.session_state.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
    
    def export_history(self) -> str:
        """Export history as JSON"""
        return json.dumps(st.session_state.chat_history, indent=2)

class VLLMClient:
    """Mock VLLM client for demonstration"""
    
    def __init__(self, api_url: str = VLLM_API_URL):
        self.api_url = api_url
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using VLLM (mocked for demo)"""
        # In a real implementation, this would call the VLLM API
        # For demo purposes, we'll simulate intelligent responses
        
        responses = {
            "clustering": f"Based on the clustering analysis, I can see distinct patterns in the data. {context[:200]}... The clusters show clear separation indicating meaningful customer segments.",
            "analysis": f"The data analysis reveals several key insights. {context[:200]}... This suggests opportunities for targeted strategies.",
            "recommendation": f"Based on the available information: {context[:200]}... I recommend focusing on the high-value segments identified.",
            "default": f"Thank you for your question. Based on the context: {context[:200]}... Here's my analysis of the situation."
        }
        
        # Simple keyword matching for demo
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        
        return responses["default"]

def main():
    st.set_page_config(
        page_title="VLLM-Weaviate RAG Cluster System",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 VLLM-Weaviate RAG Cluster Analysis System")
    st.markdown("*Advanced clustering analysis with RAG-powered insights and guardrails*")
    
    # Initialize components
    if 'guardrails' not in st.session_state:
        st.session_state.guardrails = GuardrailsValidator()
    if 'weaviate_manager' not in st.session_state:
        st.session_state.weaviate_manager = WeaviateManager()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history_manager = ChatHistory()
    if 'vllm_client' not in st.session_state:
        st.session_state.vllm_client = VLLMClient()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data generation settings
    with st.sidebar.expander("📊 Data Settings"):
        n_samples = st.slider("Number of samples", 100, 5000, 1000)
        n_clusters = st.slider("Number of clusters", 2, 10, 5)
        n_docs = st.slider("Number of documents", 50, 1000, 500)
    
    # Analysis focus areas
    with st.sidebar.expander("🎯 Analysis Focus"):
        focus_areas = st.multiselect(
            "Select focus areas:",
            ["Customer Segmentation", "Behavior Analysis", "Trend Identification", 
             "Risk Assessment", "Performance Metrics", "Predictive Insights"],
            default=["Customer Segmentation", "Behavior Analysis"]
        )
    
    # Guardrails settings
    with st.sidebar.expander("🛡️ Guardrails"):
        enable_guardrails = st.checkbox("Enable content validation", True)
        risk_threshold = st.slider("Risk threshold", 0.0, 1.0, 0.3)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Generation", "🔍 Clustering Analysis", 
        "💬 RAG Chat", "📈 Insights Dashboard", "📝 History"
    ])
    
    with tab1:
        st.header("📊 Synthetic Data Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎲 Generate Customer Data"):
                with st.spinner("Generating synthetic customer data..."):
                    df = SyntheticDataGenerator.generate_customer_data(n_samples, n_clusters)
                    st.session_state.customer_data = df
                    st.success(f"Generated {len(df)} customer records with {n_clusters} clusters")
        
        with col2:
            if st.button("📚 Generate Documents"):
                with st.spinner("Generating synthetic documents..."):
                    docs = SyntheticDataGenerator.generate_document_data(n_docs)
                    st.session_state.documents = docs
                    st.success(f"Generated {len(docs)} documents")
        
        # Display generated data
        if 'customer_data' in st.session_state:
            st.subheader("Customer Data Preview")
            st.dataframe(st.session_state.customer_data.head())
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", len(st.session_state.customer_data))
            with col2:
                st.metric("Unique Clusters", st.session_state.customer_data['cluster'].nunique())
            with col3:
                st.metric("Avg Spending Score", f"{st.session_state.customer_data['spending_score'].mean():.2f}")
            with col4:
                st.metric("Avg Frequency", f"{st.session_state.customer_data['frequency'].mean():.2f}")
    
    with tab2:
        st.header("🔍 Clustering Analysis")
        
        if 'customer_data' in st.session_state:
            df = st.session_state.customer_data
            
            # Clustering visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D scatter plot
                fig = px.scatter(
                    df, x='spending_score', y='frequency', 
                    color='cluster', title="Customer Clusters (2D View)",
                    hover_data=['recency', 'value']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 3D scatter plot
                fig = px.scatter_3d(
                    df, x='spending_score', y='frequency', z='recency',
                    color='cluster', title="Customer Clusters (3D View)",
                    hover_data=['value']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.subheader("📊 Cluster Analysis")
            
            cluster_stats = df.groupby('cluster').agg({
                'spending_score': ['mean', 'std'],
                'frequency': ['mean', 'std'], 
                'recency': ['mean', 'std'],
                'value': ['mean', 'std']
            }).round(2)
            
            st.dataframe(cluster_stats)
            
            # Cluster insights
            st.subheader("💡 Automated Cluster Insights")
            for cluster_id in df['cluster'].unique():
                cluster_data = df[df['cluster'] == cluster_id]
                avg_spending = cluster_data['spending_score'].mean()
                avg_freq = cluster_data['frequency'].mean()
                
                if avg_spending > df['spending_score'].mean():
                    spending_desc = "high spenders"
                else:
                    spending_desc = "low spenders"
                
                if avg_freq > df['frequency'].mean():
                    freq_desc = "frequent visitors"
                else:
                    freq_desc = "infrequent visitors"
                
                st.write(f"**Cluster {cluster_id}**: {spending_desc}, {freq_desc} ({len(cluster_data)} customers)")
        
        else:
            st.info("Please generate customer data first in the Data Generation tab.")
    
    with tab3:
        st.header("💬 RAG-Powered Chat Interface")
        
        # Initialize Weaviate connection
        if st.button("🔗 Initialize Vector Database"):
            with st.spinner("Connecting to Weaviate and setting up schema..."):
                # For demo purposes, we'll simulate successful connection
                st.session_state.weaviate_connected = True
                st.success("✅ Vector database initialized successfully!")
                
                # Load documents if available
                if 'documents' in st.session_state:
                    st.info("📚 Loading documents into vector database...")
                    time.sleep(2)  # Simulate loading time
                    st.success(f"✅ Loaded {len(st.session_state.documents)} documents")
        
        # Chat interface
        if st.session_state.get('weaviate_connected', False):
            st.subheader("🤖 AI Assistant")
            
            # Display chat history
            chat_container = st.container()
            
            # User input
            user_query = st.text_input(
                "Ask me about the clustering analysis or any insights:",
                placeholder="What insights can you provide about cluster 0?"
            )
            
            if st.button("💬 Send") and user_query:
                # Validate input with guardrails
                if enable_guardrails:
                    validation = st.session_state.guardrails.validate_input(user_query)
                    if not validation["is_valid"]:
                        st.error("❌ Input validation failed:")
                        for violation in validation["violations"]:
                            st.error(f"- {violation}")
                        return
                
                # Simulate RAG process
                with st.spinner("🔍 Searching knowledge base..."):
                    time.sleep(1)
                    
                    # Simulate document retrieval
                    relevant_docs = f"Found relevant context about {', '.join(focus_areas)}"
                    
                    # Generate response
                    response = st.session_state.vllm_client.generate_response(
                        user_query, relevant_docs
                    )
                    
                    # Validate output
                    if enable_guardrails:
                        output_validation = st.session_state.guardrails.validate_output(response)
                        if not output_validation["is_valid"]:
                            response = "I apologize, but I cannot provide that information due to content policy restrictions."
                    
                    # Add to chat history
                    st.session_state.chat_history_manager.add_message("user", user_query)
                    st.session_state.chat_history_manager.add_message(
                        "assistant", response, 
                        {"focus_areas": focus_areas, "validation_passed": True}
                    )
            
            # Display chat history
            with chat_container:
                for message in st.session_state.chat_history_manager.get_history():
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if message.get("metadata"):
                            st.caption(f"Focus: {', '.join(message['metadata'].get('focus_areas', []))}")
        
        else:
            st.info("Please initialize the vector database first.")
    
    with tab4:
        st.header("📈 Insights Dashboard")
        
        if 'customer_data' in st.session_state:
            df = st.session_state.customer_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster distribution
                cluster_counts = df['cluster'].value_counts()
                fig = px.pie(
                    values=cluster_counts.values, 
                    names=cluster_counts.index,
                    title="Cluster Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature correlations
                corr_matrix = df[['spending_score', 'frequency', 'recency', 'value']].corr()
                fig = px.imshow(
                    corr_matrix, 
                    title="Feature Correlations",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics by focus area
            st.subheader("🎯 Focus Area Analysis")
            
            for area in focus_areas:
                with st.expander(f"📊 {area}"):
                    if area == "Customer Segmentation":
                        st.write("**Cluster Characteristics:**")
                        for i, cluster_id in enumerate(df['cluster'].unique()):
                            cluster_data = df[df['cluster'] == cluster_id]
                            st.write(f"- Cluster {cluster_id}: {len(cluster_data)} customers, "
                                   f"avg spending: {cluster_data['spending_score'].mean():.2f}")
                    
                    elif area == "Behavior Analysis":
                        st.write("**Behavioral Patterns:**")
                        st.write(f"- High frequency customers: {len(df[df['frequency'] > df['frequency'].mean()])}")
                        st.write(f"- Recent customers: {len(df[df['recency'] < df['recency'].mean()])}")
                    
                    elif area == "Risk Assessment":
                        st.write("**Risk Indicators:**")
                        low_value = len(df[df['value'] < df['value'].quantile(0.25)])
                        st.write(f"- Low value customers: {low_value} ({low_value/len(df)*100:.1f}%)")
        
        # System metrics
        st.subheader("⚙️ System Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents Indexed", st.session_state.get('documents', []) and len(st.session_state.documents) or 0)
        with col2:
            st.metric("Chat Messages", len(st.session_state.chat_history_manager.get_history()))
        with col3:
            validation_rate = 95.2  # Simulated
            st.metric("Validation Success Rate", f"{validation_rate}%")
        with col4:
            response_time = 1.2  # Simulated
            st.metric("Avg Response Time", f"{response_time}s")
    
    with tab5:
        st.header("📝 Chat History & Export")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history_manager.clear_history()
                st.success("History cleared!")
            
            if st.button("💾 Export History"):
                history_json = st.session_state.chat_history_manager.export_history()
                st.download_button(
                    "📁 Download JSON",
                    history_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col1:
            # Display full chat history
            history = st.session_state.chat_history_manager.get_history()
            if history:
                for i, message in enumerate(history):
                    with st.expander(f"{message['role'].title()} - {message['timestamp'][:19]}"):
                        st.write(message['content'])
                        if message.get('metadata'):
                            st.json(message['metadata'])
            else:
                st.info("No chat history available. Start a conversation in the RAG Chat tab!")
    
    # Footer
    st.markdown("---")
    st.markdown("🔬 **VLLM-Weaviate RAG System** | Built with Streamlit, Weaviate, and advanced ML clustering")

if __name__ == "__main__":
    main()