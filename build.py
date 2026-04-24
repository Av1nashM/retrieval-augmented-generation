import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# ============================================
# CONFIGURATION - Adjust these parameters
# ============================================

# Chunk size: how many tokens per text chunk
# - Smaller (200-300): More precise, better for specific queries
# - Larger (800-1000): More context, better for summaries
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Number of chunks to retrieve per query
TOP_K = 5

# ============================================
# SETUP MODELS
# ============================================

print("🔧 Initializing models...")

# Set up the LLM (generation)
Settings.llm = Ollama(
    model="command-r7b",
    request_timeout=120.0,
    temperature=0.1  # Low temperature = more factual, less creative
)

# Set up the embedding model
Settings.embed_model = HuggingFaceEmbedding(
    cache_folder="./models",
    model_name="BAAI/bge-large-en-v1.5"
)

# Set up the text splitter
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ============================================
# LOAD DOCUMENTS
# ============================================

print(f"📁 Loading documents from ./data...")

if not os.path.exists("./data"):
    os.makedirs("./data")
    print("⚠️  Created ./data folder. Please add your documents there and re-run.")

documents = SimpleDirectoryReader("./data").load_data()
print(f"✅ Loaded {len(documents)} documents")

# ============================================
# BUILD VECTOR STORE
# ============================================

print(f"🔨 Building vector store with chunk_size={CHUNK_SIZE}...")

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# Save to disk for later use
index.storage_context.persist(persist_dir="./storage")
print(f"💾 Vector store saved to ./storage")
print(f"📊 Total chunks created: {len(index.docstore.docs)}")

# ============================================
# TEST THE RETRIEVAL
# ============================================

print("\n🧪 Testing retrieval with sample query...")
query = "What are the main topics covered in these documents?"
retriever = index.as_retriever(similarity_top_k=TOP_K)
results = retriever.retrieve(query)

print(f"\n📚 Retrieved {len(results)} most relevant chunks:")
for i, result in enumerate(results):
    print(f"\n--- Chunk {i+1} (Score: {result.score:.3f}) ---")
    print(f"Source: {result.metadata.get('file_name', 'Unknown')}")
    print(f"Preview: {result.text[:200]}...")
