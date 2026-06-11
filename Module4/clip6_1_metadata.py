# Store documents with metadata in Pinecone and perform filtered similarity search

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

print("Start")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langchain-metadata-demo"

# Create index if not exists
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

docs = [
    Document(
        page_content="2026 leave policy for employees in India with updated maternity benefits",
        metadata={"country": "India", "dept": "HR", "year": 2026, "type": "policy"}
    ),
    Document(
        page_content="2023 leave policy for employees in India",
        metadata={"country": "India", "dept": "HR", "year": 2023, "type": "policy"}
    ),
    Document(
        page_content="2022 leave policy for employees in India",
        metadata={"country": "India", "dept": "HR", "year": 2022, "type": "policy"}
    ),
    Document(
        page_content="2026 leave policy for employees in US",
        metadata={"country": "US", "dept": "HR", "year": 2026, "type": "policy"}
    ),
    Document(
        page_content="2026 leave policy for employees in UK",
        metadata={"country": "UK", "dept": "HR", "year": 2026, "type": "policy"}
    ),
    Document(
        page_content="Finance policy for reimbursements in India 2026",
        metadata={"country": "India", "dept": "Finance", "year": 2026, "type": "policy"}
    ),
    Document(
        page_content="Legal compliance policy for employees in India 2026",
        metadata={"country": "India", "dept": "Legal", "year": 2026, "type": "policy"}
    ),
    Document(
        page_content="Guide to employee leave application process in India",
        metadata={"country": "India", "dept": "HR", "year": 2026, "type": "guide"}
    ),
    Document(
        page_content="FAQ on leave rules for employees in India",
        metadata={"country": "India", "dept": "HR", "year": 2026, "type": "faq"}
    ),
    Document(
        page_content="Employee benefits and leave structure overview",
        metadata={"country": "India", "dept": "HR", "year": 2026, "type": "general"}
    ),
    Document(
        page_content="HR handbook for employees in India",
        metadata={"country": "India", "dept": "HR", "year": 2026, "type": "handbook"}
    ),
    Document(
        page_content="Global employee leave policies comparison",
        metadata={"country": "Global", "dept": "HR", "year": 2026, "type": "report"}
    ),
    Document(
        page_content="Technical documentation for HR systems",
        metadata={"country": "India", "dept": "Tech", "year": 2026, "type": "doc"}
    ),
]

ids = [f"d_{i:03}" for i in range(1, len(docs) + 1)]

# Add documents (indexing)
vector_store.add_documents(docs, ids=ids)
print("Documents indexed")

query = "Latest leave policy for employees in India"

# Search without metadata filtering
results = vector_store.similarity_search(query, k=3)

print("\nWithout metadata filtering:")
for doc in results:
    print(doc.page_content, "|", doc.metadata)

# Search with metadata filtering
results = vector_store.similarity_search(
    query,
    k=3,
    filter={
        "country": "India",
        "dept": "HR",
        "year": 2026,
        "type": "policy"
    }
)

print("\nWith metadata filtering:")
for doc in results:
    print(doc.page_content, "|", doc.metadata)

print("\nEnd")



# # Range filtering (year > 2022)
# results = vector_store.similarity_search(
#     query,
#     k=3,
#     filter={
#         "country": "India",
#         "year": {"$gt": 2022}
#     }
# )


# Comparison operators and logical operators to control results

# {"year": {"$gt": 2022}}   # greater than
# {"year": {"$gte": 2024}}  # greater than or equal
# {"year": {"$lt": 2023}}   # less than
# {"year": {"$lte": 2024}}  # less than or equal

# {"country": {"$ne": "US"}}  # not equal
# {"country": {"$in": ["India", "US"]}}  # match any value in the list
# {"dept": {"$nin": ["Finance"]}}  # exclude listed values

# {"$or": [{"dept": "HR"}, {"country": "US"}]} # returns documents that satisfy ANY condition (HR OR US)