# Creating and saving a FAISS vector store from sample documents using embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Sample Documents 
documents = [
    "West region customers frequently complain about delayed shipping and occasional stock shortages.",
    "Customers in the East region report product quality issues and packaging problems.",
    "West region feedback highlights strong pricing satisfaction but concerns about delivery delays.",
    "North region customers report poor customer support despite decent product durability.",
    "South region customers appreciate fast delivery and overall service quality.",
    "East region feedback indicates good sales performance but recurring quality-related complaints.",
    "Customers in the North region are happy with product durability but mention slow support responses.",
    "South region feedback consistently highlights excellent service and smooth delivery experience.",
    "Across regions, customers suggest improving tracking updates and communication.",
    "West region shows a mix of positive pricing feedback and negative delivery experiences."
]

# Create Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create FAISS Index
vectorstore = FAISS.from_texts(documents, embeddings)

# Save Locally
vectorstore.save_local("faiss_index")
print("FAISS index created and saved successfully!")