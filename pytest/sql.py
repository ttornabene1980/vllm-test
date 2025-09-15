# test_sql_agent.py
import pytest
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase

@pytest.fixture(scope="module")
def db():
    # Replace with your actual connection URI
    return SQLDatabase.from_uri(
        "postgresql+psycopg2://sorgente:sorgente@xps:5432/sorgente?sslmode=require"
    )

@pytest.fixture(scope="module")
def llm():
    # Example LLM setup, replace with your actual model
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

@pytest.fixture(scope="module")
def sql_agent(db, llm):
    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-functions",
        verbose=True
    )

def test_list_tables(sql_agent):
    """
    Test that the SQL agent can list tables in the database.
    """
    query = "List all tables in the database"
    
    try:
        # Use invoke instead of run (deprecated)
        result = sql_agent.invoke(query)
        print("Agent result:", result)
    except Exception as e:
        pytest.fail(f"SQL Agent execution failed: {e}")
