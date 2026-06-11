# Demonstrates splitting text into chunks using CharacterTextSplitter

from langchain_text_splitters import CharacterTextSplitter

text = (
    "A" * 100 + "\n\n" +
    "B" * 250 + "\n\n" +
    "C" * 600 + "\n\n" +
    "D" * 300 + "\n\n" +
    "E" * 500 + "\n\n" +
    "F" * 100
)

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=400,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    # Replace actual newlines with '\n' so they are visible
    visible_content = chunk.replace("\n", "\\n")
    length = len(chunk)
    status = "OVER LIMIT" if length > 400 else "OK"
    print(f"Chunk {i+1}: {length} characters {status}\n")
    print(f"Content: |{visible_content}|")
    print("-" * 40)
    