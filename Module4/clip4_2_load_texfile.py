# Load text file, split into chunks, store in Pinecone, and perform similarity search
 
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()

print("Start")

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Load file using TextLoader
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

print("Number of documents loaded:", len(documents))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)
print("Number of chunks created:", len(docs))

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-demo-chunks"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created")

index = pc.Index(index_name)

# Create vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print("Vector store ready")

ids = [f"chunk_{i:02}" for i in range(len(docs))]

# Add documents (indexing)
vector_store.add_documents(docs, ids=ids)
print("Documents indexed")

# Query 1
query1 = "How to stay healthy?"
results1 = vector_store.similarity_search(query1, k=2)

print("\nQuery 1 Results:")
for doc in results1:
    print("-", doc.page_content)

# Query 2
query2 = "What is deep learning used for?"
results2 = vector_store.similarity_search(query2, k=2)

print("\nQuery 2 Results:")
for doc in results2:
    print("-", doc.page_content)

print("\nEnd")