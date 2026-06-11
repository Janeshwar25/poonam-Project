# Building a hybrid pipeline combining structured SQL queries and unstructured document retrieval with routing

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///sales.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
sql_executor = QuerySQLDatabaseTool(db=db)
schema = db.get_table_info()

sql_prompt = ChatPromptTemplate.from_template("""
You are a SQL expert.

Given this database schema:
{schema}

Write a SQL query to answer:
{question}
Rules:
- Only use available tables and columns
- Do NOT guess column names
- Use valid SQLite syntax
                                              
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


grounding_prompt = ChatPromptTemplate.from_template("""
User question:                                                    
{question}

You executed the following SQL query:
{query}

The database returned this result:
{result}

Explain clearly what this result means in one short, clear sentence.
Do not describe tuple formatting.Answer the user's question using the SQL result.
Provide a clear natural language answer based only on the SQL result.
Do NOT make assumptions beyond the SQL result.
""")

structured_chain = (
    RunnablePassthrough.assign(
        query=sql_generator
    )
    | RunnablePassthrough.assign(
        query=lambda x: clean_sql(x["query"])
    )
    | RunnablePassthrough.assign(
        result=lambda x: sql_executor.invoke(x["query"])
    )
    | grounding_prompt | llm | StrOutputParser()
)

# Router (Decide query type)
router_prompt = ChatPromptTemplate.from_template("""
You are routing questions to data sources.
If the question:
- Requires numeric calculation from database → structured
- Requires document summarization or text related information → unstructured
- Requires BOTH numeric data and document insights → hybrid

If question:
- asks "why", "feedback", "issues" → unstructured
- asks numbers, totals, averages → structured
- asks BOTH → hybrid    

If the question contains multiple intents (e.g., "and", "also"), classify as hybrid.                                             

Return ONLY one word in lowercase: structured, unstructured, or hybrid
Question: {question}
""")

router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda x: x.strip().lower())
)

# Unstructured retrieval setup (Vector store)
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Branching logic based on query type
structured_input = RunnableLambda(lambda x: {"question": x["question"]})

structured_branch = RunnablePassthrough.assign(
    structured=structured_input | structured_chain,
    docs=lambda _: "No documents retrieved"
)

unstructured_branch = RunnablePassthrough.assign(
    structured=lambda _: "No structured data available",
    docs=lambda x: "\n\n".join(
        [d.page_content for d in retriever.invoke(x["question"])]
    ) or "No relevant documents found"
)

hybrid_branch = RunnablePassthrough.assign(
    structured=structured_input | structured_chain,
    docs=lambda x: "\n\n".join(
        [d.page_content for d in retriever.invoke(x["question"])]
    ) or "No relevant documents found"
)

# Final answer prompt
answer_prompt = ChatPromptTemplate.from_template("""
You are answering a user question using:

Structured data result:
{structured}

Document insights:
{docs}

Question:
{question}

Give a clear final answer combining both sources when available.
If only one source is useful, rely on that.
""")

final_chain = (
     RunnablePassthrough.assign(route=router_chain)
    | RunnableBranch(
        (lambda x: x["route"] == "structured", structured_branch),
        (lambda x: x["route"] == "unstructured", unstructured_branch),
        hybrid_branch
    )
    | RunnablePassthrough.assign(
        question=lambda x: x["question"]
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Execute the pipeline on multiple sample questions
questions = [
    "Which region has the highest total sales?",
    "What issues do customers report in the West region?",
    "What is total sales in East region and what issues do customers mention there?",
]
for q in questions:
    print("\n====================================")
    print("Question:", q)
    print("Route:", router_chain.invoke({"question": q}))
    response = final_chain.invoke({"question": q})
    print(response)