# Generate query and document embeddings using Hugging Face

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embedding for a query 
query = "What is machine learning?"

query_embedding = embeddings.embed_query(query)

print("\nEmbedding dimension:", len(query_embedding))
print("First 10 values:", query_embedding[:10])

# Generate embeddings for multiple documents
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Cooking recipes require ingredients and instructions."
]

doc_embeddings = embeddings.embed_documents(documents)

print("\nNumber of document embeddings:", len(doc_embeddings))
print("Dimension:", len(doc_embeddings[0]))

print("First 10 values of first document embedding:")
print(doc_embeddings[0][:10])
