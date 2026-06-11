# Evaluate embedding search using recall@k and precision@k

from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Query
query = "How can I improve my sleep quality?"

# Documents
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

# Ground truth (relevant documents)
relevant_docs = {
    "Tips for better sleep and healthy bedtime habits",
    "Ways to reduce stress before sleep",
    "Why sleep schedule is important",
    "Common causes of insomnia and sleep disorders"
}

# Generate embeddings
query_vec = embeddings.embed_query(query)
doc_vecs = embeddings.embed_documents(documents)

# Cosine similarity function using numpy
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm

# Compute similarity scores
results = []
for doc, vec in zip(documents, doc_vecs):
    score = cosine_similarity(query_vec, vec)
    results.append((score, doc))

# Sort documents by similarity score (highest first)
results.sort(key=lambda x: x[0], reverse=True)

print("\nRanked Results:\n")
for score, doc in results:
    print(f"{score:.3f} -> {doc}")

# Recall@K and Precision@K
k = 3  # number of top results to evaluate
top_k_docs = [doc for _, doc in results[:k]] 

# Count relevant documents in top K
relevant_found = sum(1 for doc in top_k_docs if doc in relevant_docs)

# Recall@K
recall_at_k = relevant_found / len(relevant_docs)

# Precision@K
precision_at_k = relevant_found / k

print(f"\nRecall@{k}: {recall_at_k:.2f}")
print(f"Precision@{k}: {precision_at_k:.2f}")