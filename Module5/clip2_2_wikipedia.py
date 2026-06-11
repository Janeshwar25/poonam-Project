# Retrieves top relevant Wikipedia articles for a query

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=3)

results = retriever.invoke("Football World Cup")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:300]}")
