from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Downloading embedding model...")
embed_model = HuggingFaceEmbedding(
    cache_folder="./models",
    model_name="BAAI/bge-large-en-v1.5"
)
print("Done! Model cached locally.")
