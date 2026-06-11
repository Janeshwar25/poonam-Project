# Load FAISS index and perform similarity search with relevance scores

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

print("Start")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Load FAISS vector store from disk
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True # Safe here since we created the file ourselves
)
print("Vector store loaded")

# Run query
query = "How can I improve my sleep quality?"

results = vector_store.similarity_search_with_score(query, k=3)
print("\nRaw distance scores (FAISS L2 - lower is better):")
for doc, score in results:
    print(f"{score:.4f} -> {doc.page_content}")


results = vector_store.similarity_search_with_relevance_scores(query, k=3)
print("\nNormalized relevance scores (0 to 1 - higher is better):")
for doc, score in results:
    print(f"{score:.4f} -> {doc.page_content}")

print("\nEnd")