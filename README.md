# RAG Agent Service

A modular Retrieval-Augmented Generation (RAG) service that uses MQTT messaging to ingest facts, answer questions, and manage knowledge via a vector store.  
Built with LangChain, ChromaDB, HuggingFace embeddings, and support for multiple LLM backends (Ollama, llama.cpp HTTP server, and llama.cpp binary).

## Features

- Add and store information in a ChromaDB vector store (`add_info` MQTT command)
- Forget or delete stored documents by metadata filter (`forget_info` MQTT command)
- Ask questions with context retrieval and RAG prompting (`ask_question` MQTT command)
- Session-based chat support using MQTT and LangChain chat API
- Multiple LLM backends: Ollama, llama.cpp (HTTP server and CLI)
- Embedded logging over MQTT topics for tracking operations
- QoS 1 MQTT subscriptions and publishes for reliable message delivery
- Decorator-based injection of Messager and RAGController into handlers
- Dynamic handler registration in a dedicated `handlers/` directory

## Future Features

- Tools integration (external tool calling)
- Environment RAG (environmental context retrieval)
- Agentic behavior (multi-step decision making agents)
