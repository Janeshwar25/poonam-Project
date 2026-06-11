# Adding result validation in the pipeline

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///sales.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_executor = QuerySQLDatabaseTool(db=db)
schema = db.get_table_info()

# SQL Generation Prompt
sql_prompt = ChatPromptTemplate.from_template("""
You are a SQL expert.

Given this database schema:
{schema}

Write a SQL query to answer:
{question}

Return ONLY the SQL query.
Do not include markdown formatting.
Do not use backticks.
""")

sql_generator = (
    {
        "schema": lambda _: schema,
        "question": lambda x: x["question"]
    }
    | sql_prompt
    | llm
    | StrOutputParser()
)

def clean_sql(query: str) -> str:
    query = query.strip()
    if query.startswith("```"):
        query = query.replace("```sql", "")
        query = query.replace("```", "")
    return query.strip()

def validate_sql(query: str) -> str:
    q = query.lower()

    if not q.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    forbidden = ["drop", "delete", "update", "insert", "alter"]
    if any(word in q for word in forbidden):
        raise ValueError("Unsafe SQL detected.")

    return query

checker_prompt = ChatPromptTemplate.from_template("""
You are a SQL expert.

Given the database schema, the user's question, and the SQL query,
check if the query correctly answers the question.

Database Schema:
{schema}

User Question:
{question}

SQL Query:
{query}

If the query is correct, return it unchanged.
If incorrect, return a corrected SQL query.

Return ONLY the SQL query.
Do not include explanations.
""")

sql_checker = checker_prompt | llm | StrOutputParser()

def check_sql(query: str, question: str) -> str:
    checked_query = sql_checker.invoke({
        "query": query,
        "question": question, 
        "schema": schema 
    })

    print("\n--- SQL Checker ---")
    print("Original SQL:", query)
    print("Checked SQL :", checked_query)

    return checked_query

def verify_result(result):
    # Empty result
    if not result:
        return "No data found for this query."

    # Handle NULL aggregation case like [(None,)]
    if isinstance(result, list) and result == [(None,)]:
        return "No data found for this query."

    # Large result (limit number of rows)
    if isinstance(result, list) and len(result) > 20:
        return "Result is too large. Please refine your query."

    # Return result as string for consistent LLM input
    return str(result)

# Final Answer Prompt 
answer_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using the SQL result.

Rules:
- Do NOT make assumptions beyond the SQL result.
- Include the SQL query and the SQL result and then provide the final answer.
- Keep the answer clear and concise.

Output format:

User Question:
{question}

SQL Query Used:
{query}

SQL Result:
{result}

Final Answer:
Provide a clear natural language answer based only on the SQL result.
""")

# Full pipeline
full_chain = (
    RunnablePassthrough.assign(
        query=sql_generator
    )
    | RunnablePassthrough.assign(
        query=lambda x: clean_sql(x["query"])
    )
    | RunnablePassthrough.assign(
        query=lambda x: validate_sql(x["query"])
    )
    | RunnablePassthrough.assign(
        query=lambda x: check_sql(x["query"], x["question"])
    )
    | RunnablePassthrough.assign(
        result=lambda x: sql_executor.invoke(x["query"])
    )
    | RunnablePassthrough.assign(
        result=lambda x: verify_result(x["result"])
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Execute the pipeline on multiple sample questions
questions = [
    "What is total sales in West region?",
    "What is total sales in Central region?",
    "What is the average sales in East region?"
]

for q in questions:
    print("\n====================================")
    print("Question:", q)

    try:
        response = full_chain.invoke({"question": q})
        print(response)
    except Exception as e:
        print("Error:", str(e))