import dspy
import sqlite3
from dotenv import load_dotenv

from tools import execute_sql, get_schema, save_data_to_csv


# --- DSPy Agent Definition ---
class SQLAgentSignature(dspy.Signature):
    """
    You are an expert SQL agent that answers user questions in natural language by querying a SQLite database.
    Tools:
    - execute_sql(query: str): Executes SQL, returns results or error.
    - get_schema(table_name: str or None): Returns table list or columns/types.
    - save_data_to_csv(data: list of rows, filename: str): Saves to CSV.
    Rules:
    - Start with get_schema.
    - Use execute_sql for queries.
    - Fix errors in retries (max 7).
    - Save large results to CSV, report path.
    - Natural language final answer only from data.
    """

    question = dspy.InputField(desc="The user's natural language question.")
    initial_schema = dspy.InputField(desc="The initial database schema to guide you.")
    answer = dspy.OutputField(
        desc="The final, natural language answer to the user's question."
    )


class SQLAgent(dspy.Module):
    """The SQL Agent Module"""
    def __init__(self, tools: list[dspy.Tool]):
        super().__init__()
        # Initialize the ReAct agent.
        self.agent = dspy.ReAct(
            SQLAgentSignature,
            tools=tools,
            max_iters=7,
        )

    def forward(self, question: str, initial_schema: str) -> dspy.Prediction:
        """The forward pass of the module."""
        result = self.agent(question=question, initial_schema=initial_schema)
        return result


def configure_llm():
    """Configures the DSPy language model."""
    load_dotenv()
    llm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=4000)
    dspy.settings.configure(lm=llm)

    print("[Agent] DSPy configured with gpt-4o-mini model.")
    return llm


def create_agent(conn: sqlite3.Connection, query_history: list[str] | None = None) -> dspy.Module | None:
    if not configure_llm():
        return

    execute_sql_tool = dspy.Tool(
        name="execute_sql",
        desc="Executes SQL query. Input: query (str). Output: rows string or error.",
        func=lambda query: execute_sql(conn, query, query_history),
    )

    get_schema_tool = dspy.Tool(
        name="get_schema",
        desc="Gets schema. Input: table_name (str/None). Output: table names list or [(col, type)].",
        func=lambda table_name: get_schema(conn, table_name),
    )

    save_csv_tool = dspy.Tool(
        name="save_data_to_csv",
        desc="Saves data to CSV. Input: data (list of tuples/lists), filename (str). Output: success path or error.",
        func=save_data_to_csv
    )

    all_tools = [execute_sql_tool, get_schema_tool, save_csv_tool]


    # 2. Instantiate and run the agent
    agent = SQLAgent(tools=all_tools)

    return agent
