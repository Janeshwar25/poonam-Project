# Generate query and document embeddings using OpenAI

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embedding for a query 
query_embedding = embeddings.embed_query( "What is machine learning?")
print("\nEmbedding dimension:", len(query_embedding))
print("First 10 values of query embedding:")
print(query_embedding[:10])

# Generate embeddings for multiple documents
documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Cooking recipes require ingredients and instructions."
]

doc_embeddings = embeddings.embed_documents(documents)
print("\nNumber of document embeddings:", len(doc_embeddings))
print("Dimension of each embedding:", len(doc_embeddings[0]))

print("First 10 values of first document embedding:")
print(doc_embeddings[0][:10])
