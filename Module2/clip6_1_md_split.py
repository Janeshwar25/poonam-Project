# Splits a markdown document into structured chunks based on header hierarchy using MarkdownHeaderTextSplitter

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# Machine Learning Guide

## Introduction
Machine learning is a field of artificial intelligence that focuses on building systems that learn from data. 
These systems improve automatically as they are exposed to more data over time. 
The core idea behind machine learning is to enable computers to recognize patterns and make predictions without 
being explicitly programmed for every scenario.

Machine learning algorithms are widely used in many real-world applications. 
For example, recommendation systems analyze user preferences and suggest products or movies. 
Email services use machine learning models to detect spam messages and filter them automatically.

Healthcare organizations use machine learning models to assist doctors in diagnosing diseases. 
Financial institutions use it to detect fraudulent transactions by identifying unusual patterns in large datasets. 
Autonomous vehicles also rely heavily on machine learning to interpret sensor data and make driving decisions.

Because of these capabilities, machine learning has become one of the most important technologies in modern computing.

## Types of Learning
There are several types of machine learning approaches used in practice.

### Supervised Learning
In supervised learning, models are trained using labeled data.

### Unsupervised Learning
In unsupervised learning, the algorithm discovers patterns without labeled data.

## Applications
Machine learning is used in healthcare, finance, recommendation systems, autonomous vehicles, and many other domains.
"""

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

docs = markdown_splitter.split_text(markdown_text)

for i, doc in enumerate(docs, 1):
    print(f"\nChunk {i}")
    print("Metadata:", doc.metadata)
    print(doc.page_content)

