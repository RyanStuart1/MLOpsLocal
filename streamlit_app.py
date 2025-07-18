import streamlit as st
from pipeline.chatbot import show_chatbot_sidebar

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="📊")
st.title("📊 Welcome to the Credit Risk App")

st.markdown("""
This dashboard helps you:
- Predict loan default risk
- View model performance metrics
- Explore insights about the dataset and model
- Stream and Batch Model monitoring 

Use the sidebar to navigate between pages.
""")

show_chatbot_sidebar()