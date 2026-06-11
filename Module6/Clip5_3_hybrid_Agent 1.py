# Building a hybrid agent combining safe SQL querying and document retrieval tools

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///sales.db", sample_rows_in_table_info=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create SQL toolkit (original tools)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Find original SQL execution tool
base_sql_tool = [t for t in tools if t.name == "sql_db_query"][0]

@tool
def safe_sql_query(query: str) -> str:
    """
    Execute ONLY safe SELECT SQL queries on the database.
    """
    q = query.lower()

    # Allow only SELECT
    if not q.strip().startswith("select"):
        return "Error: Only SELECT queries are allowed."

    # Block dangerous operations
    forbidden = ["drop", "delete", "update", "insert", "alter"]
    if any(word in q for word in forbidden):
        return "Error: Unsafe SQL detected."

    # Execute safe query
    return base_sql_tool.invoke(query)

# Replace only execution tool
safe_tools = []
for t in tools:
    if t.name == "sql_db_query":
        safe_tools.append(safe_sql_query)  # replace unsafe tool
    else:
        safe_tools.append(t)  # keep others

# Unstructured retrieval setup (Vector store)
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Add a retrieve tool 
@tool
def retrieve_documents(question: str) -> str:
    """Retrieve customer feedback or document-based insights."""
    docs = retriever.invoke(question)
    return "\n\n".join([d.page_content for d in docs]) or "No relevant documents found"

safe_tools.append(retrieve_documents)

# Create agent
agent = create_agent(
    model=llm,
    tools=safe_tools,
    system_prompt="""
You are a hybrid data assistant.

Rules for SQL:
- ALWAYS check the database schema using tools before writing any SQL query.
- Use ONLY tables and columns that exist in the schema.
- Do NOT guess table or column names.

CRITICAL:
- ALWAYS use the SQL tool result as the source of truth.
- NEVER use values from schema previews or sample rows.
- If a SQL tool returns a result, use that result exactly.

Tool usage:
- Use SQL tools for numeric/database queries
- Use retrieve_documents tool for customer feedback, issues, or summaries
- Use both tools if needed

General rules:
- Always use tools, do not guess
- Do not perform unsafe SQL operations
- Do not make up answers
"""
)

# Execute the agent and display tool usage along with final answers
questions = [
    "Which region has the highest total sales?",
    "What issues do customers report in the West region?",
    "What is total sales in East region and what issues do customers mention there?",
]

# Execute the agent and display tool usage along with final answers
for q in questions:
    print("====================================")
    print("Question:", q)

    response = agent.invoke({"messages": [{"role": "user", "content": q}]})
    # Print tool usage
    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"Tool: {tool_call['name']:<20} Arguments: {tool_call['args']}")

    print("\nFinal Answer:")
    print(response["messages"][-1].content)


# # Execute the agent and display tool usage along with results and final answers
# for q in questions:
#     print("====================================")
#     print("Question:", q)

#     response = agent.invoke({"messages": [{"role": "user", "content": q}]})

#     for msg in response["messages"]:

#         # Tool calls (what agent decided)
#         if hasattr(msg, "tool_calls") and msg.tool_calls:
#             for tool_call in msg.tool_calls:
#                 print(f"Tool: {tool_call['name']:<20} Arguments: {tool_call['args']}")

#         # Tool result (what tool returned)
#         if isinstance(msg, ToolMessage):
#             print(f"{'Tool Result: '} {msg.content}")

#     print("\nFinal Answer:")
#     print(response["messages"][-1].content)

    
# # Stream step-by-step agent execution
# for step in agent.stream({"messages": [{"role": "user", "content": "What is the average sales in East region?"}]}):
#     print("\n---")
#     print(step)
    

