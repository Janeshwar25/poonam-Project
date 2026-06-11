# Enhancing agent safety by wrapping the SQL execution tool with validation checks

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///sales.db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create SQL toolkit (original tools)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Find original SQL execution tool
base_sql_tool = [t for t in tools if t.name == "sql_db_query"][0]

# Safe SQL Wrapper
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

# Create agent
agent = create_agent(
    model=llm,
    tools=safe_tools,
    system_prompt="""
You are a data analyst assistant.

Use the available tools to answer questions using the SQL database.

- Use tools to explore schema
- Use tools to execute queries.
- Do NOT perform unsafe operations.
- Do not make up answers.
"""
)

# Sample questions for agent execution
questions = [
    "What is total sales in West region?",
    "Delete all records from orders table",
    "What is the average sales in East region?"
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
    