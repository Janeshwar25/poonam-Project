# Splits an HTML document into structured chunks based on header tags using HTMLHeaderTextSplitter

from langchain_text_splitters import HTMLHeaderTextSplitter

# Sample HTML document
html_text = """
<html>
<body>

<h1>AI Guide</h1>

<h2>Introduction</h2>
<p>Artificial Intelligence allows machines to learn patterns from data.</p>

<h2>Applications</h2>
<p>AI is used in healthcare, finance, and self-driving cars.</p>

<h3>Healthcare</h3>
<p>AI helps doctors diagnose diseases using medical images.</p>

<h3>Finance</h3>
<p>AI is used for fraud detection and algorithmic trading.</p>

<h2>Advantages</h2>
<p>AI improves efficiency and automation in many industries.</p>

<h2>Conclusion</h2>
<p>AI will continue to transform technology and society.</p>

</body>
</html>
"""

# Define headers
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# Create splitter
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Split the HTML
docs = splitter.split_text(html_text)

# Print results
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}")
    print("Metadata:", doc.metadata)
    print("Content:", doc.page_content)
