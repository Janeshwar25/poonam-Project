# Compare similarity search and MMR search using a Pinecone vector store

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("Start")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-mmr-demo"

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created")

index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print("Vector store ready")

# Create documents
docs = [
    Document(page_content="Vector databases are used in search engines"),
    Document(page_content="Vector databases improve semantic search"),
    Document(page_content="Vector databases power modern search systems"),
    Document(page_content="Vector databases are used in recommendation systems"),
    Document(page_content="Vector databases are widely used in recommendation systems like Netflix and Amazon"),
    Document(page_content="Vector databases help recommend products and content"),
    Document(page_content="Vector databases are applied in fraud detection"),
    Document(page_content="Vector databases help detect anomalies in transactions"),
    Document(page_content="Fraud detection systems use vector databases"),
    Document(page_content="Vector databases are used in image search"),
    Document(page_content="Vector databases enable similarity search for images"),
    Document(page_content="Image retrieval systems rely on vector databases"),
]

ids = [f"d_{i:03}" for i in range(1, len(docs) + 1)]

# Add documents (indexing)
vector_store.add_documents(docs, ids=ids)
print("Documents indexed")

# Query
query = "Applications of vector databases"

# Similarity Search
print("\n--- Similarity Search Results ---\n")

results_sim = vector_store.similarity_search(query, k=4)

for i, doc in enumerate(results_sim, 1):
    print(f"{i}. {doc.page_content}")

# MMR Search
print("\n--- MMR Search Results ---\n")

results_mmr = vector_store.max_marginal_relevance_search(
    query,
    k=4,
    fetch_k=10
)

for i, doc in enumerate(results_mmr, 1):
    print(f"{i}. {doc.page_content}")

