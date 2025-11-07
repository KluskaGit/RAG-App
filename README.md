# RAG-App

My own Retrieval-Augmented Generation (RAG) application built with Python, featuring a Streamlit-based chat interface and ChromaDB vector storage. This application enables intelligent document querying using embeddings and LLMs.

## Features

- **Multi-Provider Embedding Support**: Compatible with 25+ embedding providers including OpenAI, Ollama, HuggingFace, Cohere, and more
- **Flexible Vector Storage**: Supports both local and cloud ChromaDB instances
- **Smart Document Processing**: PDF document loading with automatic text splitting and chunking
- **Interactive Chat Interface**: Built with Streamlit for seamless user interaction
- **Configurable LLM Integration**: Works with various LLM providers through OpenAI-compatible API
- **Token-Based Text Splitting**: Intelligent document chunking using tiktoken
- **Document Metadata Tracking**: Maintains source and page information for citations

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Supported Embedding Providers](#supported-embedding-providers)
- [Architecture](#architecture)
- [Development](#development)

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- [ChromaDB](https://trychroma.com) instance (local or cloud)
- [Ollama](https://ollama.com/) (if using Ollama embeddings)
- API keys for your chosen embedding and LLM providers

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/KluskaGit/RAG-App.git
cd RAG-App
```

2. **Install dependencies with uv** (skip if you are using Docker)
```bash
uv sync
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
RETRIEVER_API_KEY=your_retriever_api_key
OPENAI_API_KEY=your_huggingface_token
CHROMA_CLOUD_API_KEY=your_chroma_cloud_api_key
TENANT_ID = your_tenant_id
```

4. **Configure the application**

Edit `appconfig.yaml` to match your setup (see [Configuration](#configuration) section).

5. **Add documents**

Place your PDF documents in the `data/` folder for indexing.

## Configuration

The application is configured via `appconfig.yaml`. Here's a breakdown of the configuration options:

### Retriever Configuration (Embeddings)

Arguments in retriever section must match function arguments of chosen provider. You can find them in [chroma embedding functions](https://docs.trychroma.com/docs/embeddings/embedding-functions)

```yaml
retriever:
  provider: ollama  # Choose from 25+ providers
  url: http://localhost:11434  # Ollama server URL
  model_name: jeffh/intfloat-multilingual-e5-large-instruct:f32
  # api_key: RETRIEVER_API_KEY  # Read from environment variable
```

### LLM Configuration

```yaml
LLM:
  model: openai/gpt-oss-20b:nebius
  base_url: https://router.huggingface.co/v1
  api_key: OPENAI_API_KEY  # Read from environment variable
```

### Vector Store Configuration

```yaml
vectorstore:
  chroma_client: local  # 'local' or 'cloud'
  collection_name: documents
  
  # For local ChromaDB
  host: localhost
  port: 8000
  
  # For ChromaDB Cloud
  # tenant: TENANT_ID  # Optional
  database: rag-vectorstore
  api_key: CHROMA_CLOUD_API_KEY
```

## Usage

### Starting the Application (Locally)

1. **Run ChromaDB locally** (if using local mode):
```bash
uv run chroma run
```
2. **Run Ollama locally**
```bash
ollama pull <model name>
ollama serve
```

4. **Start the Streamlit app**:
```bash
uv run streamlit run main.py
```

### Starting the Application in Docker
```bash
docker compose up -d
```

### Using the Chat Interface

1. The application will automatically index all PDF documents in the `data/` folder on startup
2. Type your question in the chat input
3. The system will retrieve relevant document chunks and generate a contextual response
4. Source documents and page numbers are displayed with each answer

### Adding New Documents

1. Place PDF files in the `data/` folder
2. Restart the application to index new documents

## Project Structure

```
RAG-App/
├── main.py                      # Application entry point
├── appconfig.yaml               # Configuration file
├── pyproject.toml              # Project dependencies
├── .env                        # Environment variables (create this)
├── data/                       # PDF documents to index
├── chroma/                     # Local ChromaDB storage
├── rag/                        # Core RAG functionality
│   ├── __init__.py
│   ├── pipeline.py             # Main RAG pipeline orchestration
│   ├── loaders.py              # Document loading (PDF support)
│   ├── splitters.py            # Token-based text splitting
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding.py        # Multi-provider embedding factory
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── document.py         # Document schema definitions
│   └── vectorstores/
│       ├── __init__.py
│       ├── vectorstore.py      # Base vector store interface
│       ├── chroma_local.py     # Local ChromaDB implementation
│       └── chroma_cloud.py     # Cloud ChromaDB implementation
└── ui/
    ├── __init__.py
    └── chat.py                 # Streamlit chat interface
```

## Supported Embedding Providers

The application supports 25+ embedding providers:

- **OpenAI** - OpenAI embeddings
- **Ollama** - Local models via Ollama
- **HuggingFace** - HuggingFace models
- **Cohere** - Cohere embeddings
- **Google** - Google PaLM, Vertex AI, Generative AI
- **Instructor** - Instructor embeddings
- **Jina** - Jina AI embeddings
- **Sentence Transformers** - Local sentence transformers
- **Amazon Bedrock** - AWS Bedrock
- **Mistral** - Mistral AI
- **VoyageAI** - VoyageAI embeddings
- **Together AI** - Together AI embeddings
- **And many more...**

See `rag/embeddings/embedding.py` for the complete list and documentation.

## Architecture

### RAG Pipeline Flow

1. **Document Ingestion**
   - PDFs are loaded from the `data/` folder
   - Text is extracted page by page with metadata

2. **Text Processing**
   - Documents are split into chunks using token-based splitting
   - Chunk size: 500 tokens, overlap: 150 tokens
   - Each chunk is hashed for deduplication

3. **Embedding & Storage**
   - Chunks are embedded using the configured provider
   - Embeddings are stored in ChromaDB with metadata

4. **Query Processing**
   - User query is embedded using the same model
   - Similarity search retrieves relevant chunks
   - Context is formatted for the LLM

5. **Response Generation**
   - LLM receives context and user query
   - Generates response based on retrieved documents
   - Returns answer with source citations

### Key Components

- **FileLoader**: Handles PDF document loading
- **TokenTextSplitter**: Splits documents into manageable chunks
- **Embedding Functions**: Provides unified interface for 25+ providers
- **Vector Stores**: Manages ChromaDB (local or cloud)
- **Pipeline**: Orchestrates the entire RAG workflow
- **Chat UI**: Streamlit-based interactive interface

## Development

### Text Splitting Configuration

Modify chunk size and overlap in `rag/pipeline.py`:
```python
self.splitter = TokenTextSplitter(chunk_size=500, overlap=150)
```

### Adding New Document Types

Extend `rag/loaders.py` to support additional file formats:
```python
def load_txt(self, path: str) -> list[Document]:
    # Implement text file loading
    pass
```

### Custom Embedding Providers

The embedding factory automatically maps providers. Add new providers in `rag/embeddings/embedding.py` if needed.

### Vector Store Implementations

Create custom vector store implementations by extending the base interface in `rag/vectorstores/`.

## Dependencies

Key dependencies:
- `chromadb` - Vector database
- `streamlit` - Web interface
- `openai` - LLM API client
- `pypdf` - PDF processing
- `tiktoken` - Token counting
- `ollama` - Local LLM support
- `pydantic` - Data validation

See `pyproject.toml` for complete dependency list.

### Package Management

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### ChromaDB Connection Issues
- Ensure ChromaDB is running on the specified host/port
- Check your API keys in `.env` file
- Verify network connectivity for cloud instances

### Embedding Errors
- Confirm your embedding provider is properly configured
- Check API keys are set correctly
- Verify model names match available models

### Document Loading Issues
- Ensure PDFs are not corrupted or password-protected
- Check file permissions in the `data/` folder
- Verify sufficient disk space for indexing

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Python, ChromaDB, and Streamlit**