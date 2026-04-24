from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# ============================================
# CONFIGURATION
# ============================================

TOP_K = 5
TEMPERATURE = 0.1

# ============================================
# IMPORTANT: SET UP SETTINGS FIRST
# These must be configured BEFORE loading the index
# ============================================

print("🔧 Configuring models...")

# Set up the LLM (generation)
Settings.llm = Ollama(
    model="command-r7b",
    temperature=TEMPERATURE,
    request_timeout=120.0
)

# Set up the embedding model - THIS MUST MATCH what you used in build.py
Settings.embed_model = HuggingFaceEmbedding(
    cache_folder="./models",
    model_name="BAAI/bge-large-en-v1.5"
)

print("✅ Models configured")

# ============================================
# LOAD STORED INDEX
# ============================================

print("🔄 Loading existing vector store...")

# Check if storage exists
import os
if not os.path.exists("./storage"):
    print("❌ No storage found! Please run 'python build.py' first.")
    exit(1)

# Load the saved vector store
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

print(f"✅ Loaded index")

# ============================================
# SETUP RETRIEVAL AND QUERY ENGINE
# ============================================

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=TOP_K,
)

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    llm=Settings.llm  # Explicitly pass the LLM
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# ============================================
# INTERACTIVE QUERY LOOP
# ============================================

print("\n" + "="*50)
print("💬 RAG Assistant Ready!")
print("="*50)
print("Ask questions about your documents. Type 'exit' to quit.")
print("Type 'scores' to see similarity scores with responses.")
print("-"*50)

show_scores = False

while True:
    query = input("\n🔍 Your question: ").strip()
    
    if query.lower() == 'exit':
        print("👋 Goodbye!")
        break
    elif query.lower() == 'scores':
        show_scores = not show_scores
        print(f"📊 Score display: {'ON' if show_scores else 'OFF'}")
        continue
    elif not query:
        continue
    
    print("🤔 Thinking...")
    
    try:
        # Get the response with retrieved chunks
        response = query_engine.query(query)
        
        print(f"\n📝 Answer: {response.response}")
        
        if show_scores and hasattr(response, 'source_nodes'):
            print("\n📚 Source Chunks (by relevance):")
            for i, source in enumerate(response.source_nodes):
                score = getattr(source, 'score', 0.0)
                print(f"\n--- Chunk {i+1} (Similarity: {score:.3f}) ---")
                print(f"Source: {source.metadata.get('file_name', 'Unknown')}")
                print(f"Text: {source.text[:300]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try re-running 'python build.py' to rebuild the vector store.")
