# Building an LCEL pipeline for natural language to SQL generation, execution, and answer synthesis

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
You are an SQL expert.

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

# Final Answer Prompt 
answer_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using the SQL result.

Rules:
- Do NOT make assumptions beyond the SQL result.
- Include the SQL query and the SQL result and then provide the final answer.
- Keep the answer clear and concise.

Output format:
User Question: {question}
SQL Query Used: {query}
SQL Result: {result}
Final Answer: Provide a clear natural language answer based only on the SQL result.
""")

# Full Pipeline
full_chain = (
    RunnablePassthrough.assign(
        query=sql_generator
    )
    | RunnablePassthrough.assign(
        query=lambda x: clean_sql(x["query"])
    )
    | RunnablePassthrough.assign(
        result=lambda x: sql_executor.invoke(x["query"])
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Execute the pipeline on multiple sample questions
questions = [
    "What is total sales in West region?",
    "What is the average sales in East region?",
    "Show total sales grouped by region.",
    "How many orders are there in the West region?",
    "Which region has the highest total sales?"
]

for q in questions:
    print("====================================")
    response = full_chain.invoke({"question": q})
    print(response)
