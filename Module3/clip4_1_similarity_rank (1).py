# Rank documents by similarity to a query using embeddings

from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Query
query = "How can I improve my sleep quality at night?"

documents = [
    "Penguins live in Antarctica",
    "Tips for better sleep and healthy bedtime habits",              
    "The solar system has eight planets",
    "Ways to reduce stress before sleep",                            
    "How to cook pasta",
    "Meditation helps with relaxation",                             
    "Why sleep schedule is important",                               
    "Exercise improves overall health",                              
    "How to stay productive during the day",
    "Common causes of insomnia and sleep disorders",                 
    "Stress at work affects productivity",
    "Healthy diet improves energy levels"
]

# Convert text into embeddings
query_vector = embeddings.embed_query(query)
doc_vectors = embeddings.embed_documents(documents)

# Cosine similarity function using numpy
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm

# Compute similarity scores
results = []
for doc, vec in zip(documents, doc_vectors):
    score = cosine_similarity(query_vector, vec)
    results.append((score, doc))

# Sort documents by similarity score (highest first)
results.sort(key=lambda x: x[0], reverse=True)

print("\nSimilarity scores:\n")
for score, doc in results:
    print(f"{score:.3f} -> {doc}")

# Show top 3 matches
print("\nTop 3 most relevant documents:\n")
for score, doc in results[:3]:
    print(f"{score:.3f} -> {doc}")

