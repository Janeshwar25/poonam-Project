# Retrieves top documents using BM25Retriever

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

docs = [
    Document(page_content="The football world cup is held every four years"),
    Document(page_content="Python is a programming language"),
    Document(page_content="Argentina won the football world cup in 2022"),
    Document(page_content="Cricket is a bat and ball game"),
    Document(page_content="Basketball is played on a court with a hoop"),
    Document(page_content="The weather is nice today"),
    Document(page_content="I like to eat apples"),
]

retriever = BM25Retriever.from_documents(docs, k=2)

results = retriever.invoke("football world cup")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
