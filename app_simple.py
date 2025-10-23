"""
Simple Streamlit App for Testing Railway Deployment
"""

import streamlit as st
import os

# Streamlit App Configuration
st.set_page_config(
    page_title="Chat with the Map - Test",
    page_icon="🗺️",
    layout="wide"
)

# App Title
st.title("🗺️ Chat with the Map - Test Version")
st.markdown("Testing Railway deployment...")

# Check environment variables
st.header("🔧 Environment Check")

env_vars = {
    "PGDATABASE": os.getenv('PGDATABASE', 'Not set'),
    "PGUSER": os.getenv('PGUSER', 'Not set'),
    "PGHOST": os.getenv('PGHOST', 'Not set'),
    "PGPORT": os.getenv('PGPORT', 'Not set'),
    "OLLAMA_URL": os.getenv('OLLAMA_URL', 'Not set'),
    "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY', 'Not set')
}

for var, value in env_vars.items():
    if value != 'Not set':
        st.success(f"✅ {var}: {value}")
    else:
        st.warning(f"⚠️ {var}: Not set")

# Test database connection
st.header("🗄️ Database Test")

try:
    import psycopg2
    
    db_params = {
        'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
        'user': os.getenv('PGUSER', 'dhanush'),
        'password': os.getenv('PGPASSWORD', ''),
        'host': os.getenv('PGHOST', 'localhost'),
        'port': int(os.getenv('PGPORT', 5432))
    }
    
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    st.success(f"✅ Database connected successfully!")
    st.info(f"PostgreSQL version: {version}")
    
    # Check if counties table exists
    cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'counties');")
    table_exists = cur.fetchone()[0]
    
    if table_exists:
        st.success("✅ Counties table exists!")
        cur.execute("SELECT COUNT(*) FROM counties;")
        count = cur.fetchone()[0]
        st.info(f"Counties in database: {count}")
    else:
        st.warning("⚠️ Counties table does not exist. Run migrate_data.py to create it.")
    
    cur.close()
    conn.close()
    
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")

# Simple chat interface
st.header("💬 Test Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type a test message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = f"✅ Test successful! You wrote: '{prompt}'"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("**This is a test version to verify Railway deployment is working correctly.**")
