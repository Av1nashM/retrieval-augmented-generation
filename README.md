# 📚 Local RAG Assistant

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using AI. No cloud APIs, no data leaving your computer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/ollama-local-green.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Privacy First](https://img.shields.io/badge/privacy-first-purple.svg)](https://en.wikipedia.org/wiki/Privacy)

## ✨ Features

- **100% Local** - Runs entirely on your laptop using Ollama and open-source models
- **Private** - Your documents never leave your machine
- **Document Support** - PDF, TXT, PPT, PPTX, Markdown files
- **Conversation Memory** - Remembers context within a chat session
- **Simple File Management** - Drag and drop documents into the `data/` folder
- **Interactive CLI** - Ask questions and get answers from your documents

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Command-R7B (via Ollama) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector Store | ChromaDB (via LlamaIndex) |
| Framework | LlamaIndex |
| Language | Python 3.10+ |

## 📁 Project Structure
local-rag-assistant/
├── data/ # 📂 Place your documents here
├── storage/ # 💾 Vector database (auto-generated)
├── models/ # 🤖 Cached embedding models
├── build.py # 🔨 Builds vector store from documents
├── query.py # 💬 Query without memory
├── chat.py # 🧠 Chat with conversation memory
├── requirements.txt # 📦 Python dependencies
└── README.md # 📖 This file


## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Installation

1. **Clone or create the project folder**
   ```bash
   mkdir local-rag-assistant
   cd local-rag-assistant

2. **Create virtual environment**
   ```bash
   python -m venv venv

3. **Activate virtual environment**
  - Windows:
   ```bash
   venv\Scripts\activate
```
  - MacOS/Linux
  ```bash
  source venv/bin/activate
```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Ollama**

6. **Pull required models**
   ```bash
   ollama pull command-r7b
   ollama pull nomic-embed-text
   ```

7. **Add documents**
   - Drag and drop PDFs, TXTs, or PPTs into the data/ folder using File Explorer.

