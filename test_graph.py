import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import psycopg2
import json
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langchain_community.llms import Ollama # Deprecated
from langchain_ollama import OllamaLLM as Ollama # New import
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import operator

load_dotenv()

# Mock Streamlit
class MockSt:
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def write(self, msg): print(f"WRITE: {msg}")

st = MockSt()

# Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = "llama3.2:3b"
DB_PARAMS = {
    'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
    'user': os.getenv('PGUSER', 'dhanush'),
    'password': os.getenv('PGPASSWORD', ''),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', 5432))
}

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sql_query: str
    query_result: str
    map_data: dict
    error: str

# Define Nodes (Simplified from app.py)
def sql_agent(state):
    print("--- SQL AGENT ---")
    messages = state['messages']
    question = messages[-1].content
    
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)
    
    prompt = f"""
    You are a PostGIS expert.
    Database: USCountyDB
    Table: counties (geom, statefp, countyfp, name, state_name, state_abbr, ...)
    
    Question: {question}
    
    Return ONLY the SQL query. No markdown, no explanation.
    IMPORTANT: BORDER/NEIGHBOR QUERIES
    - If the user asks for "bordering", "adjacent", "next to", "neighbors", "touching" of a county:
    - YOU MUST USE A SELF-JOIN on the counties table!
    - Pattern: SELECT c2.name, c2.state_name, ST_AsGeoJSON(c2.geom) as geom_geojson, ST_Y(ST_Centroid(c2.geom)) as centroid_lat, ST_X(ST_Centroid(c2.geom)) as centroid_lon FROM counties c1, counties c2 WHERE c1.name ILIKE '%[Target]%' AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id
    - NOTE: The 'name' column usually does NOT contain "County" (e.g. 'Los Angeles'). Use wildcards (e.g. '%Los Angeles%') or remove "County" from the target name.
    
    IMPORTANT: If the user asks to "show", "map", "visualize", or "display", YOU MUST SELECT GEOMETRY COLUMNS:
    ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon
    
    IMPORTANT: Use ILIKE for string comparisons to be case-insensitive (e.g. state_name ILIKE 'california').
    State names are stored as Title Case (e.g. 'California', 'New York').

    IMPORTANT: AGGREGATION & MAPS
    - If the user asks for "count", "how many", "total area", or "sum" for a region (e.g. "counties in California", "area of Texas"):
    - DO NOT use COUNT() or SUM() in SQL!
    - INSTEAD, SELECT the individual rows (name, state, area, geom) so they can be mapped!
    - Python will calculate the total/count for you.
    - Example: "How many counties in CA?" -> SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson... FROM counties WHERE state_abbr='CA'
    - Example: "Total area of Texas?" -> SELECT name, state_name, aland, ST_AsGeoJSON(geom) as geom_geojson... FROM counties WHERE state_name='Texas'
    """
    
    try:
        response = llm.invoke(prompt)
        sql = response.strip().replace('```sql', '').replace('```', '').strip()
        print(f"Generated SQL: {sql}")
        return {"sql_query": sql, "messages": [AIMessage(content=sql)]}
    except Exception as e:
        return {"error": str(e)}

def execute_query(state):
    print("--- EXECUTE QUERY ---")
    sql = state['sql_query']
    if not sql:
        return {"error": "No SQL generated"}
        
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        print(f"Rows found: {len(rows)}")
        return {"query_result": str(rows)}
    except Exception as e:
        print(f"Query failed: {e}")
        return {"error": str(e)}

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("sql_agent", sql_agent)
workflow.add_node("execute_query", execute_query)

workflow.set_entry_point("sql_agent")
workflow.add_edge("sql_agent", "execute_query")
workflow.add_edge("execute_query", END)

app = workflow.compile()

# Run Test
print("Running Graph Test...")
inputs = {"messages": [HumanMessage(content="Which counties border Los Angeles County?")]}
try:
    result = app.invoke(inputs)
    print("Final Result:", result)
except Exception as e:
    print("Graph Execution Failed:", e)
