# Builds a custom compression retriever using LCEL to filter relevant information

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(temperature=0)

docs = [
    Document(page_content="The capital of Japan is Tokyo. It is famous for its technology and culture."),
    Document(page_content="The capital of India is New Delhi. It is known for its rich history."),
    Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower."),
]

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Get top 2

# Compression Logic (The "Extractor")
compression_prompt = ChatPromptTemplate.from_template("""
Extract ONLY the parts of the document relevant to the query. 
If the document is not relevant, return "NO_RELEVANT_INFO".

Query: {query}
Document: {document}
""")

compression_chain = compression_prompt | llm | StrOutputParser()

def compress_docs(inputs):
    query = inputs["query"]
    docs = inputs["docs"]
    compressed = []

    for d in docs:
        # Extract relevant content for each doc
        result = compression_chain.invoke({
            "query": query,
            "document": d.page_content
        }).strip()

        # Only append if the LLM found actual relevant info
        if result and "NO_RELEVANT_INFO" not in result:
            compressed.append(result)

    return compressed

# The LCEL "Retriever" Chain
compression_retriever_chain = (
    {
        "docs": base_retriever,
        "query": RunnablePassthrough()
    }
    | RunnableLambda(compress_docs)
)

query = "What is the capital of France?"

# Base retriever (for comparison)
base_results = base_retriever.invoke(query)

print("\n--- Raw documents (From Vector Store) ---")
for d in base_results:
    print("-", d.page_content)

# Compression chain
compressed_results = compression_retriever_chain.invoke(query)

print("\n--- Compressed documents (After LLM Extraction) ---")
for r in compressed_results:
    print("-", r)
