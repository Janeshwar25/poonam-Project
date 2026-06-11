# Loads a text file and splits it into chunks with metadata using CharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../data/my_files/sports.txt", encoding="utf-8")
documents = loader.load()

print("Number of documents (pages):", len(documents))

# Create splitter
splitter = CharacterTextSplitter(
    separator="",  
    chunk_size=800,
    chunk_overlap=0
)

# Split documents
split_docs = splitter.split_documents(documents)

print("\nTotal chunks created:", len(split_docs))

# Display chunk details
for i, chunk in enumerate(split_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Chunk size: {len(chunk.page_content)} characters")
    print(f"Chunk metadata: ", chunk.metadata)

    visible_content = chunk.page_content.replace("\n", "\\n")
    print(f"Chunk content: |{visible_content}|") 