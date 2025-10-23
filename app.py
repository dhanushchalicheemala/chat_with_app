
"""
Chat with the Map - Streamlit Application with LangGraph ReAct Agent
Natural Language Interface for US Counties PostGIS Database
"""

import streamlit as st
import subprocess
import psycopg2
import json
import requests
import os
from typing import Tuple, List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Database configuration - Railway will provide these as environment variables
DB_PARAMS = {
    'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
    'user': os.getenv('PGUSER', 'dhanush'),
    'password': os.getenv('PGPASSWORD', ''),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', 5432))
}

# Ollama configuration - Railway service URL
OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = "llama3.2:3b"

# Schema information for the LLM
SCHEMA_INFO = """
Table: counties
Columns:
- name (county name like 'Los Angeles')
- state_name (full state name like 'California')
- state_abbr (state code like 'CA')
- aland (land area in square meters)
- awater (water area in square meters)
- geom (geometry - use ST_Area(geom::geography)/1000000 for km²)
- intptlat, intptlon (latitude and longitude of center point)

Important functions:
- ST_Area(geom::geography)/1000000 for area in km²
- ST_Distance(geom1::geography, geom2::geography)/1000 for distance in km
- ST_Touches(geom1, geom2) to find neighboring counties
- ST_Contains(geom1, geom2) for containment

Example queries:
- SELECT COUNT(*) FROM counties WHERE state_name = 'California';
- SELECT name, ST_Area(geom_4326::geography)/1000000 as area_km2 FROM counties ORDER BY area_km2 DESC LIMIT 5;
- For spatial operations, use geom_4326 (SRID 4326) for compatibility with geography functions
"""

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

@tool
def convert_nl_to_sql(question: str) -> str:
    """
    Convert a natural language question about US counties to a PostgreSQL SQL query.
    
    Args:
        question: The natural language question about US counties
        
    Returns:
        A PostgreSQL SQL query string
    """
    prompt = f"""{SCHEMA_INFO}

    Question: {question}

    Generate ONLY the PostgreSQL SQL query, no explanation:"""

    try:
        # Try Ollama first (if available)
        if os.getenv('OLLAMA_URL'):
            try:
                result = subprocess.run(
                    ['ollama', 'run', MODEL_NAME, prompt],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                sql = result.stdout.strip()
                sql = sql.replace('```sql', '').replace('```', '').strip()
                return sql
            except:
                pass
        
        # Fallback to OpenAI if Ollama is not available
        if os.getenv('OPENAI_API_KEY'):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            response = llm.invoke(prompt)
            sql = response.content.strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            return sql
        
        # Final fallback - return a simple query
        return "SELECT COUNT(*) FROM counties LIMIT 1;"

    except Exception as e:
        return f"SELECT COUNT(*) FROM counties LIMIT 1;"

@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the US counties database and return the results.
    
    Args:
        sql_query: The SQL query to execute
        
    Returns:
        A formatted string with the query results
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        cur.execute(sql_query)

        # Get column names
        columns = [desc[0] for desc in cur.description] if cur.description else []

        # Fetch results
        rows = cur.fetchall()

        cur.close()
        conn.close()

        if not rows:
            return "No results found."

        # Format results
        if len(columns) == 1 and columns[0] == 'count':
            return f"**Result:** {rows[0][0]}"

        result_text = f"**Found {len(rows)} result(s):**\n\n"

        for i, row in enumerate(rows[:10], 1):  
            if len(columns) == 1:
                result_text += f"{i}. {row[0]}\n"
            else:
                row_data = ", ".join([f"{col}: {val}" for col, val in zip(columns, row)])
                result_text += f"{i}. {row_data}\n"

        if len(rows) > 10:
            result_text += f"\n*...and {len(rows) - 10} more*"

        return result_text

    except Exception as e:
        return f"Database connection error: {e}. Please check if the database is properly configured and the counties table exists."

@tool
def get_database_schema() -> str:
    """
    Get information about the database schema for US counties.
    
    Returns:
        A string describing the database schema
    """
    return SCHEMA_INFO

# =============================================================================
# LANGGRAPH AGENT SETUP
# =============================================================================

def create_agent():
    """Create a simple agent that uses tools directly"""
    
    def simple_agent(question: str) -> str:
        """Simple agent that converts NL to SQL and executes it"""
        try:
            # Step 1: Convert natural language to SQL
            sql_query = convert_nl_to_sql.invoke({"question": question})
            
            # Step 2: Execute the SQL query
            results = execute_sql_query.invoke({"sql_query": sql_query})
            
            # Step 3: Format the response
            response = f"""I'll help you answer that question about US counties.

**Generated SQL Query:**
```sql
{sql_query}
```

**Results:**
{results}

Is there anything else you'd like to know about US counties?"""
            
            return response
            
        except Exception as e:
            return f"I encountered an error while processing your question: {e}. Please try rephrasing your question."
    
    return simple_agent

# Initialize the agent
@st.cache_resource
def get_agent():
    """Get or create the agent (cached for performance)"""
    return create_agent()

# =============================================================================
# AGENT EXECUTION FUNCTIONS
# =============================================================================

def run_agent(question: str) -> str:
    """Run the agent with a user question and return the response"""
    try:
        agent = get_agent()
        return agent(question)
    except Exception as e:
        return f"Error running agent: {e}"

# Streamlit App Configuration
st.set_page_config(
    page_title="Chat with the Map",
    page_icon="🗺️",
    layout="wide"
)

# App Title
st.title("🗺️ Chat with the Map")
st.markdown("Ask questions about US Counties in natural language! Powered by LangGraph ReAct Agent with Llama 3.2-3B")

# Sidebar with example questions
with st.sidebar:
    st.header("💡 Example Questions")
    example_questions = [
        "How many counties are in California?",
        "What are the 5 largest counties by area?",
        "Which counties border Los Angeles County?",
        "What is the total area of Texas?",
        "List all counties in New York",
        "Show me counties with area greater than 10,000 km²"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question}"):
            st.session_state.user_input = question

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about US counties..."):

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking and using tools..."):
            try:
                # Run the agent
                response = run_agent(prompt)
                
                # Display the response
                st.markdown(response)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Handle example question clicks
if hasattr(st.session_state, 'user_input'):
    prompt = st.session_state.user_input
    del st.session_state.user_input
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking and using tools..."):
            try:
                # Run the agent
                response = run_agent(prompt)
                
                # Display the response
                st.markdown(response)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

