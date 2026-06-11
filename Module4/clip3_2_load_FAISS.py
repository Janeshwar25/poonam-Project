# Load FAISS vector store from disk and perform similarity search on a query

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API keys
load_dotenv()

print("Start")

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Embeddings ready")

# Load FAISS vector store from disk
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True # Safe here since we created the file ourselves
)

print("Vector store loaded")

# Run query
query = "How can I improve my sleep quality?"

results = vector_store.similarity_search(query, k=3)

print("\nTop results:")
for doc in results:
    print(doc.page_content)

print("\nEnd")