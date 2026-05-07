
import sys
import os
torch_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_path):
    os.add_dll_directory(torch_path)
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from core.config import UPLOAD_DIR

st.set_page_config(page_title="DocForge AI", layout="wide")
st.title("🔨 DocForge AI ")
st.caption("Intelligent Document Analysis & Research")

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ["Upload Documents", "Query Documents"])

if page == "Upload Documents":
    exec(open("app/pages/1_Upload_Docs.py", encoding="utf-8").read(), globals())
else:
    exec(open("app/pages/2_Query_Research.py", encoding="utf-8").read(), globals())