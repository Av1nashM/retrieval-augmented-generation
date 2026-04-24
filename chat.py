from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine

# ============================================
# CONFIGURE MODELS (Same as before)
# ============================================

print("🔧 Configuring models...")

Settings.llm = Ollama(
    model="command-r7b",
    temperature=0.1,
    request_timeout=120.0
)

Settings.embed_model = HuggingFaceEmbedding(
    cache_folder="./models",
    model_name="BAAI/bge-large-en-v1.5"
)

print("✅ Models configured")

# ============================================
# LOAD INDEX
# ============================================

print("🔄 Loading vector store...")

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

print("✅ Index loaded")

# ============================================
# CREATE CHAT ENGINE WITH MEMORY
# ============================================

# Memory buffer stores last 10 exchanges
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

# Chat engine automatically rephrases questions using conversation history
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=index.as_query_engine(),
    condense_question_prompt=None,  # Uses default prompt
    memory=memory,
    llm=Settings.llm,
    verbose=True  # Shows you when it's rephrasing questions
)

# ============================================
# INTERACTIVE CHAT LOOP
# ============================================

print("\n" + "="*50)
print("💬 RAG Chat Assistant WITH Memory!")
print("="*50)
print("I remember our conversation. Type 'exit' to quit.")
print("Type 'reset' to clear memory and start over.")
print("-"*50)

while True:
    user_input = input("\n🧑 You: ").strip()
    
    if user_input.lower() == 'exit':
        print("👋 Goodbye!")
        break
    elif user_input.lower() == 'reset':
        memory.reset()
        print("🧹 Memory cleared. Starting fresh.")
        continue
    elif not user_input:
        continue
    
    print("🤔 Thinking...")
    
    try:
        response = chat_engine.chat(user_input)
        print(f"\n🤖 Assistant: {response.response}")
    except Exception as e:
        print(f"❌ Error: {e}")

