# Retrieves Wikipedia context and uses an LLM to generate an answer based on it

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

retriever = WikipediaRetriever(top_k_results=3)

llm = ChatOpenAI(model="gpt-4o-mini")

query = "Tell me about the recent Football World Cup"

# Retrieve
results = retriever.invoke(query)

# Combine context
context = "\n\n".join([doc.page_content for doc in results])

# Ask LLM
response = llm.invoke(f"Answer based on this context:\n{context}\n\nQuestion: {query}")

print(response.content)
