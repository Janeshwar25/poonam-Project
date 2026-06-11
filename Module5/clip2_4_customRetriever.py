# Implements a custom retriever that returns documents using simple keyword matching
    
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class MyCustomRetriever(BaseRetriever):
    data: List[str]  

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = []
        query_lower = query.lower()

        for text in self.data:
            if query_lower in text.lower():
                results.append(Document( page_content=text,metadata={"source": "custom_data"}))
        return results


# Sample data
data = [
    "The server status is running smoothly",
    "Database connection failed yesterday",
    "System status is currently stable",
    "Network latency is high"
]

# Create retriever
retriever = MyCustomRetriever(data=data)

# Query
docs = retriever.invoke("status")

for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")