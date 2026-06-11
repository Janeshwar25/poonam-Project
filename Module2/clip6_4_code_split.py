# Splits Python and C code into structured chunks using language-aware RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
def calculate_average(numbers):
    total = 0
    count = 0
    for n in numbers:
        total += n
        count += 1
    average = total / count
    return average


def find_max(numbers):
    max_value = numbers[0]
    for n in numbers:
        if n > max_value:
            max_value = n
    difference = max_value - numbers[0]
    return max_value, difference


def normalize_data(numbers):
    avg = calculate_average(numbers)
    normalized = []
    for n in numbers:
        value = n - avg
        normalized.append(value)
    return normalized


# main program
data = [10, 20, 30, 40]

print(calculate_average(data))
print(find_max(data))
print(normalize_data(data))
"""

c_code = """
#include <stdio.h>

int add(int a, int b) {
    int result = a + b;
    return result;
}

int subtract(int a, int b) {
    int result = a - b;
    return result;
}

int multiply(int a, int b) {
    int result = a * b;
    return result;
}

int divide(int a, int b) {
    if (b == 0) {
        return 0;
    }
    int result = a / b;
    return result;
}

int main() {
    int x = 10;
    int y = 5;

    int a = add(x, y);
    int b = subtract(x, y);
    int c = multiply(x, y);
    int d = divide(x, y);

    return 0;
}
"""
# Python splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20
)

# C splitter
c_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.C,
    chunk_size=200,
    chunk_overlap=20
)

python_chunks = python_splitter.split_text(python_code)
c_chunks = c_splitter.split_text(c_code)

print("\n--- Python Code Chunks ---")
for i, chunk in enumerate(python_chunks, 1):
    print(f"\nChunk {i}")
    print(chunk)

print("\n--- C Code Chunks ---")
for i, chunk in enumerate(c_chunks, 1):
    print(f"\nChunk {i}")
    print(chunk)