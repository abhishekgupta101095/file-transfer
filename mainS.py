import streamlit as st
from pages import filters, custom_agent

st.set_page_config(page_title="My App", page_icon="🚀")

option = st.sidebar.radio("Choose an option", ["Choose Filters (Recommended)", "Use Custom Agent (Experimental)"])

# Use conditional statements to render the corresponding page based on the option
if option == "Choose Filters (Recommended)":
    filters.app()
elif option == "Use Custom Agent (Experimental)":
    custom_agent.app()
