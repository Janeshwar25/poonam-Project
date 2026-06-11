# Full RAG pipeline to answer questions using retrieved context and an LLM

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
print("Start")

# Load story file
loader = TextLoader("story.txt", encoding="utf-8")
documents = loader.load()
print("Documents loaded:", len(documents))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)
print("Chunks created:", len(docs))

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)
print("Vector store ready")

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context:
{context}

Question:
{question}
"""
)

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL chain
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions
query = "Who is Arjun's best friend??"
response = chain.invoke(query)
print("\nQuestion:", query)
print("\nAnswer:", response)

query = "How many brothers does Arjun have?"
response = chain.invoke(query)
print("\nQuestion:", query)
print("\nAnswer:", response)

query = "Why did Arjun decide to destroy the stone?"
response = chain.invoke(query)
print("\nQuestion:", query)
print("\nAnswer:", response)

print("\nEnd")