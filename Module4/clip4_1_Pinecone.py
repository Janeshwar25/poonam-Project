# Pinecone vector store: create index, add, search, delete, and update documents

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()

print("Start")

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo"

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
documents = [
    Document(page_content="Penguins live in Antarctica"),
    Document(page_content="Tips for better sleep and healthy bedtime habits"),
    Document(page_content="The solar system has eight planets"),
    Document(page_content="Ways to reduce stress before sleep"),
    Document(page_content="Cook pasta in 5 minutes"),
    Document(page_content="Meditation helps with relaxation"),
    Document(page_content="Why sleep schedule is important"),
    Document(page_content="Exercise improves overall health"),
    Document(page_content="How to stay productive during the day"),
    Document(page_content="Common causes of insomnia and sleep disorders"),
    Document(page_content="Stress at work affects productivity"),
    Document(page_content="Healthy diet improves energy levels")
]

ids = [f"d{i:03}" for i in range(1, len(documents) + 1)]  # Generate list of IDs in the format d001, d002, d003

# Add documents 
vector_store.add_documents(documents=documents, ids=ids)
print("Documents indexed")

# Similarity search
query = "How can I improve my sleep quality?"

results = vector_store.similarity_search(query, k=3)
print("\nTop results:")
for doc in results:
    print(doc.page_content)

# Delete a document
vector_store.delete(ids=["d004"])
print("\nDeleted d004")

# Search again after deletion
results = vector_store.similarity_search(query, k=3)
print("\nResults after deletion:")
for doc in results:
    print(doc.page_content)

# Update a document (delete + re-add)
vector_store.delete(ids=["d005"])

updated_doc = Document(page_content="Good sleep and regular exercise are essential for a healthy life.")

vector_store.add_documents(documents=[updated_doc], ids=["d005"])
print("\nUpdated d005")

# Final search
results = vector_store.similarity_search(query, k=3)
print("\nFinal results after update:")
for doc in results:
    print(doc.page_content)

print("\nEnd")  