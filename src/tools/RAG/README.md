# RAG Tool (Knowledge Base)

The RAG Tool provides semantic information storage and retrieval capabilities for the Argentic framework. It uses vector embeddings to store and search through text-based knowledge, making it ideal for contextual information, user preferences, documentation, and any textual data that needs to be semantically searchable.

## Features

### Core Capabilities

- **Semantic Storage**: Text stored with vector embeddings for intelligent retrieval
- **Multi-Collection Support**: Organize knowledge into different collections
- **Contextual Retrieval**: Find relevant information based on semantic similarity
- **Metadata Management**: Rich metadata support for advanced filtering
- **CRUD Operations**: Complete Create, Read, Update, Delete functionality

### Embedding Support

- **Configurable Models**: Support for different embedding models via HuggingFace
- **Optimized Retrieval**: MMR (Maximal Marginal Relevance) and similarity search
- **Vector Storage**: ChromaDB backend for efficient vector operations

## Usage

### Actions

#### Remind (Retrieve Information)

Find relevant information based on a query using semantic similarity.

```python
action = "remind"
query = "What are the user's preferences for coffee?"
collection_name = "user_preferences"  # optional
```

#### Remember (Store Information)

Add new information to the knowledge base with automatic embedding generation.

```python
action = "remember"
content_to_add = "User prefers dark roast coffee in the morning and green tea in the afternoon"
collection_name = "user_preferences"  # optional
source = "conversation"  # optional metadata
metadata = {"category": "preferences", "importance": "high"}  # optional
```

#### Forget (Remove Information)

Remove information from the knowledge base using metadata filters.

```python
action = "forget"
where_filter = {"source": "outdated_manual"}  # required for safety
collection_name = "documentation"  # optional
```

#### List Collections

Get information about available knowledge collections.

```python
action = "list_collections"
```

## Configuration

The tool is configured via `rag_config.yaml`:

```yaml
# Embedding model configuration
embedding:
  model_name: thenlper/gte-small
  device: cpu
  normalize: true

# Vector storage settings
vector_store:
  base_directory: ./pc_knowledge_db
  default_collection: local_info

# Collection-specific settings
collections:
  local_info:
    retriever:
      k: 4 # Number of documents to retrieve
      search_type: mmr # Maximal Marginal Relevance
      fetch_k: 20 # Candidates for MMR
      distance_metric: cosine

  documentation:
    retriever:
      k: 6
      search_type: mmr
      fetch_k: 30
      distance_metric: cosine

  code_examples:
    retriever:
      k: 5
      search_type: similarity
      fetch_k: 20
      distance_metric: cosine

# Default retriever settings
default_retriever:
  k: 4
  search_type: mmr
  n_neighbors: 10
  distance_metric: cosine
  fetch_k: 20
```

## Use Cases

### Personal Assistant

- Store user preferences and habits
- Remember important dates and events
- Track personal information and context

### Documentation System

- Store technical documentation
- Enable natural language queries for help
- Maintain searchable knowledge base

### Conversation Memory

- Remember previous conversations
- Maintain context across sessions
- Store important insights and decisions

### Code Documentation

- Store code examples and snippets
- Maintain API documentation
- Track coding patterns and best practices

### Learning and Training

- Store educational content
- Build searchable course materials
- Track learning progress and notes

## Data Structure

### Document Storage

```python
{
    "page_content": "The actual text content",
    "metadata": {
        "source": "conversation|manual|import|...",
        "timestamp": 1703123456.789,
        "collection": "local_info",
        "category": "user_preference",
        "importance": "high",
        # ... custom metadata
    }
}
```

### Query Results

```python
[
    {
        "page_content": "User prefers dark roast coffee...",
        "metadata": {
            "source": "conversation",
            "collection": "user_preferences",
            "timestamp": 1703123456.789
        }
    },
    # ... more results
]
```

### Collection Statistics

```python
{
    "collections": ["local_info", "documentation", "code_examples"],
    "default_collection": "local_info",
    "count": 3
}
```

## Advanced Features

### Metadata Filtering

Filter documents during forget operations:

```python
# Remove all documents from a specific source
where_filter = {"source": "outdated_docs"}

# Remove documents older than a certain date
where_filter = {"timestamp": {"$lt": 1703000000}}

# Remove by category
where_filter = {"category": "temporary"}

# Combine filters
where_filter = {
    "source": "import",
    "category": "draft"
}
```

### Search Types

**MMR (Maximal Marginal Relevance)**:

- Reduces redundancy in results
- Better diversity in retrieved documents
- Recommended for most use cases

**Similarity Search**:

- Pure similarity-based retrieval
- Faster but may return similar documents
- Good for specific, focused queries

### Collection Management

Different collections for different purposes:

- `local_info`: Personal and contextual information
- `documentation`: Technical documentation and manuals
- `code_examples`: Code snippets and programming examples
- `conversations`: Chat history and important exchanges
- `preferences`: User settings and preferences

## Integration Examples

### With Chat Systems

```python
# Remember important information from conversation
action = "remember"
content_to_add = "User's project deadline is next Friday"
collection_name = "user_context"
metadata = {"type": "deadline", "urgency": "high"}

# Retrieve relevant context for responses
action = "remind"
query = "What deadlines does the user have?"
collection_name = "user_context"
```

### With Documentation Systems

```python
# Store API documentation
action = "remember"
content_to_add = "The /api/users endpoint accepts GET requests and returns user data..."
collection_name = "documentation"
metadata = {"type": "api", "endpoint": "/api/users"}

# Query for help
action = "remind"
query = "How do I get user information from the API?"
collection_name = "documentation"
```

### With Learning Systems

```python
# Store learning material
action = "remember"
content_to_add = "Python list comprehensions: [x for x in range(10) if x % 2 == 0]"
collection_name = "code_examples"
metadata = {"language": "python", "topic": "list_comprehensions"}

# Find examples
action = "remind"
query = "Show me Python list comprehension examples"
collection_name = "code_examples"
```

## Running the Service

Start the RAG tool service:

```bash
python src/services/rag_tool_service.py
```

The service will:

1. Initialize HuggingFace embeddings model
2. Connect to ChromaDB for vector storage
3. Connect to the messaging system (MQTT)
4. Register the knowledge_base_tool with the agent
5. Listen for knowledge operations

## Error Handling

The tool includes comprehensive error handling:

- **Empty Content**: Warns when trying to store empty text
- **Invalid Timestamps**: Automatically uses current time for invalid timestamps
- **Missing Collections**: Creates collections automatically when needed
- **Retrieval Errors**: Returns empty results with logging on failures
- **Forget Safety**: Requires non-empty filters to prevent accidental deletion

## Performance Considerations

### Embedding Model

- **Model Choice**: Smaller models (gte-small) for speed, larger for accuracy
- **Device**: Use GPU if available for faster embedding generation
- **Batch Processing**: Embeddings are generated per document

### Vector Store

- **ChromaDB**: Persistent storage with good performance
- **Collection Size**: Monitor collection sizes for optimal performance
- **Search Parameters**: Tune `k` and `fetch_k` based on use case

### Memory Usage

- **Embedding Cache**: Models cache embeddings in memory
- **Result Limits**: Configure appropriate `k` values to limit memory usage
- **Collection Cleanup**: Regular cleanup of old/irrelevant documents

## Integration

The RAG Tool integrates with:

- **Chat Systems**: Contextual conversation memory
- **Documentation**: Searchable knowledge bases
- **Learning Platforms**: Educational content storage
- **Personal Assistants**: Preference and habit tracking
- **Code Systems**: Documentation and example storage
- **Planning Systems**: Project and task information

## Best Practices

1. **Collection Organization**: Use separate collections for different types of information
2. **Metadata Strategy**: Include rich metadata for better filtering and organization
3. **Content Quality**: Store meaningful, well-formatted text for better retrieval
4. **Regular Cleanup**: Remove outdated information to maintain relevance
5. **Query Optimization**: Use specific queries for better semantic matching
6. **Source Tracking**: Always include source information for traceability
