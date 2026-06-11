# Uses MultiQueryRetriever to generate multiple queries and improve retrieval results

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

# Sample documents
docs = [
    Document(page_content="If your laptop is not turning on, check the power supply and battery."),
    Document(page_content="Slow laptop performance can be improved by closing background applications."),
    Document(page_content="WiFi issues on laptops can be caused by outdated drivers."),
    Document(page_content="Restarting the router can fix internet connection problems."),
    Document(page_content="Battery or charger faults may stop your laptop from starting."),
    Document(page_content="Close background apps to improve laptop performance."),
    Document(page_content="Check network adapter settings if WiFi is not connecting."),
    Document(page_content="Laptop overheating can lead to system shutdowns."),
    Document(page_content="Check if airplane mode is enabled if WiFi is not working."),
    Document(page_content="Network issues can often be fixed by resetting router settings.")
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

query = "WiFi not working on laptop"

results = multi_retriever.invoke(query)

print("Final Retrieved Documents:")
for doc in results:
    print("-", doc.page_content)

print("\nEnd")