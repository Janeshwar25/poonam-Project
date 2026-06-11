# Implements a custom multi-query retriever by generating multiple queries and merging unique results
 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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

# Vector store + retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(temperature=0)

# Query generation prompt
prompt = ChatPromptTemplate.from_template("""
Generate 3 different versions of the following query.
Return each query on a new line.

Query: {query}
""")

def clean_queries(text: str):
    return [ q.strip().lstrip("1234567890.- ") for q in text.split("\n") if q.strip() ]

generate_queries_chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_queries)
)

def get_unique_documents(queries, retriever):
    all_docs = []
    for query in queries:
        all_docs.extend(retriever.invoke(query))
    unique = {doc.page_content: doc for doc in all_docs}
    return list(unique.values())

retrieval_chain = RunnableLambda(
    lambda queries: get_unique_documents(queries, base_retriever)
)

multi_query_chain = generate_queries_chain | retrieval_chain

query = "WiFi not working on laptop"

results = multi_query_chain.invoke({"query": query})

print("\nFinal Retrieved Documents:")
for doc in results:
    print("-", doc.page_content)

print("\nEnd")