import streamlit as st

st.title("🚀 Railway Test App")
st.write("If you can see this, Railway deployment is working!")

st.header("Environment Check")
import os
st.write(f"Python version: {os.sys.version}")

st.write("✅ App is running successfully!")
