# Loads a text file and splits it into overlapping chunks using RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("../data/my_files/sports.txt", encoding="utf-8")
documents = loader.load()

print("Number of documents (pages):", len(documents))

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""] ,
    chunk_size=450,
    chunk_overlap=100
)

split_docs = splitter.split_documents(documents)

print("\nTotal chunks created:", len(split_docs))

# Display chunk details
for i, chunk in enumerate(split_docs):
    print(f"\n--- Chunk {i+1} ---")
    visible_content = chunk.page_content.replace("\n", "\\n")
    print(f"{visible_content}") 
