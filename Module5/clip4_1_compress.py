# Uses ContextualCompressionRetriever to filter and shorten retrieved content

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [ Document(page_content="The capital of Japan is Tokyo. It is famous for its technology and culture."),
         Document(page_content="The capital of India is New Delhi. It is known for its rich history."),
         Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower."),
]

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Get top 2

# Create the Document Compressor
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# Initialize the ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

query = "What is the capital of France?"

results = base_retriever.invoke(query)
print("Result from base retriever")
for doc in results:
   print("-", doc.page_content.strip())

compressed_results = compression_retriever.invoke(query)
print("\nResult from ContextualCompressionRetriever")
for doc in compressed_results:
    print("-", doc.page_content.strip())