import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Rare Disease CDSS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode / Medical Theme)
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4A90E2; 
        font-weight: 600;
    }
    
    /* Card/Box Styling */
    .stCard {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2D333B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    /* Primary Button */
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }

    /* Disclaimer Section */
    .disclaimer {
        font-size: 0.8rem;
        color: #888;
        padding: 10px;
        border-top: 1px solid #333;
        margin-top: 50px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80) 
    st.title("Rare Gen - CDSS")
    st.caption("AI-Powered Differential Diagnosis Assistant")
    
    st.markdown("---")
    
    # API Key Configuration
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google AI Studio API Key if not in .env")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("### Settings")
    model_temp = st.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.2, 0.1)
    retrieval_k = st.slider("Retrieved Documents (k)", 1, 10, 5)

    st.markdown("---")
    st.info("⚠️ For educational & research use only. Not a replacement for professional medical advice.")

# --- Initialization ---
@st.cache_resource
def load_resources():
    try:
        # Load Embedding Model (Local/Free)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        
        # Load Vector Store (ChromaDB)
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("cdss_knowledge_base")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create Index from Vector Store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        return index
    except Exception as e:
        st.error(f"Error initializing knowledge base: {e}")
        return None

# Load Index
index = load_resources()

# --- Main Interface ---
st.title("Rare Disease Differential Diagnosis")
st.markdown("### 🔍 Patient Case Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    symptoms = st.text_area(
        "Enter Patient Symptoms & History:",
        height=150,
        placeholder="e.g., 30yo male, recurring fever, lymphadenopathy, negative malaria test, rash on trunk..."
    )

    analyze_btn = st.button("Analyze Case", type="primary")

with col2:
    st.markdown("#### Supported Resources")
    st.markdown("- ✅ PubMed Abstracts (Rare Diseases)")
    st.markdown("- ✅ Clinical Guidelines")
    st.markdown("- ❌ No Live Web Search (Offline/Vector DB)")

# --- Analysis Logic ---
if analyze_btn and symptoms and index:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Please provide a Google Gemini API Key in the sidebar or .env file.")
    else:
        with st.spinner("Consulting knowledge base & synthesizing differential diagnosis..."):
            try:
                # 1. Setup Retrieval
                llm = Gemini(
                model="models/gemini-pro",
                temperature=model_temp,
                api_key=os.getenv("GOOGLE_API_KEY")
                )

                Settings.llm = llm
                
                retriever = VectorIndexRetriever(index=index, similarity_top_k=retrieval_k)
                query_engine = RetrieverQueryEngine(retriever=retriever)
                
                # 2. Retrieve Documents First (for display)
                retrieved_nodes = retriever.retrieve(symptoms)
                
                # 3. Generate Response
                # We construct a custom prompt to simulate a "Second Opinion"
                prompt = (
                    "You are an expert Clinical Decision Support System specializing in rare diseases.\n"
                    "Analyze the following patient symptoms and history.\n"
                    "Use the provided context from medical literature to suggest a differential diagnosis.\n"
                    "\n"
                    f"Patient Case:\n{symptoms}\n"
                    "\n"
                    "Instructions:\n"
                    "1. Provide top 3-5 potential differential diagnoses, ranked by likelihood.\n"
                    "2. Explain the reasoning for each, citing specific symptoms.\n"
                    "3. Reference the provided medical literature context where relevant.\n"
                    "4. Suggest specific next steps (labs, imaging, genetic tests).\n"
                    "5. Be concise, professional, and purely objective.\n"
                )
                
                response = query_engine.query(prompt)
                
                # --- Result Display ---
                st.success("Analysis Complete")
                
                # Using columns for result layout
                r_col1, r_col2 = st.columns([2, 1.2])
                
                with r_col1:
                    st.markdown("### 🩺 Differential Diagnosis Report")
                    st.markdown(response.response)
                
                with r_col2:
                    st.markdown("### 📚 Evidence / Retrieved Context")
                    for i, node in enumerate(retrieved_nodes):
                        with st.expander(f"Reference {i+1} (Relevance: {node.score:.2f})"):
                            st.caption(node.text[:500] + "...")
                            st.markdown("**Source:** PubMed / Guidelines")

            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

# --- Footer ---
st.markdown('<div class="disclaimer">CDSS-RAG System | Powered by LlamaIndex, Gemini 1.5 & Streamlit</div>', unsafe_allow_html=True)
