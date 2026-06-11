# Query Pinecone vector store and return similarity results with scores

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

print("Start")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo"
index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print("Vector store ready")

# Similarity search
query = "How can I improve my sleep quality?"

results = vector_store.similarity_search_with_score(query, k=3)

print("\nRaw similarity scores (cosine similarity - higher is better):")
for doc, score in results:
    print(f"{score:.4f} -> {doc.page_content}")


results = vector_store.similarity_search_with_relevance_scores(query, k=3)

print("\nNormalized relevance scores (0 to 1 - higher is better):")
for doc, score in results:
    print(f"{score:.4f} -> {doc.page_content}")

print("\nEnd")