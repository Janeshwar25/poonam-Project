# Demonstrates different retriever strategies like top-k, filtering, MMR, and threshold on a vector store

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


docs = [
    Document(
        page_content="Sleep improves brain function and memory",
        metadata={"topic": "sleep", "category": "health"}
    ),
    Document(
        page_content="Good sleep habits improve sleep quality",
        metadata={"topic": "sleep", "category": "health"}
    ),
    Document(
        page_content="Insomnia affects sleep quality and health",
        metadata={"topic": "sleep", "category": "health", "condition": "insomnia"}
    ),
    Document(
        page_content="How to cook pasta perfectly",
        metadata={"topic": "cooking", "category": "food"}
    ),
    Document(
        page_content="Deep sleep is important for recovery",
        metadata={"topic": "sleep", "category": "health"}
    ),
    Document(
        page_content="Meditation reduces stress and improves relaxation",
        metadata={"topic": "meditation", "category": "mental_health"}
    ),
    Document(
        page_content="Stress management helps improve sleep",
        metadata={"topic": "stress", "category": "mental_health"}
    ),
    Document(
        page_content="The solar system has eight planets",
        metadata={"topic": "space", "category": "science"}
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(docs, embeddings)

print("\n--- Default Retriever ---")
retriever = vectorstore.as_retriever()
results = retriever.invoke("How to improve sleep quality?")
for doc in results:
    print("-", doc.page_content)


print("\n--- Top K Retriever ---")
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2   # k is 4 by default
    }   
)
results = retriever.invoke("How to improve sleep quality?")
for doc in results:
    print("-", doc.page_content)


print("\n--- Metadata Retriever ---")
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": {"category": "health"}    
    }
)
results = retriever.invoke("How to improve sleep quality?")
for doc in results:
    print("-", doc.page_content)


print("\n--- Maximum Marginal Relevance(MMR) Retriever ---")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2,
        "fetch_k": 6
    }
)
results = retriever.invoke("How to improve sleep quality?")
for doc in results:
    print("-", doc.page_content)


print("\n--- Threshold Retriever ---")
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,                 # Return up to 5 docs
        "score_threshold": 0.3  # But only if their score is > 0.3
    }
)
results = retriever.invoke("How to improve sleep quality?")
for doc in results:
    print("-", doc.page_content)