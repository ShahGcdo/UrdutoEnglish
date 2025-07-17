import streamlit as st
try:
    import transformers
    st.success("✅ transformers is installed!")
except Exception as e:
    st.error(f"❌ transformers not installed: {e}")
