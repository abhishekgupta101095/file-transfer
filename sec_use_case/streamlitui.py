import streamlit as st
from streamlit_chat import message
import requests
import json

st.title("Intelligent Data Collection bot")
st.sidebar.title("Solution Panel")

user_query_input = st.chat_input('Enter the queries')

if user_query_input:
    message((user_query_input),is_user=True)
    inputs = {"user_query": user_query_input}
    res = requests.post(url = "http://127.0.0.1:8092/sapfin", data = json.dumps(inputs))
    st.sidebar.write(res.json()['solution'])
    try:
        st.table(res.json()['final_answer'])
    except:
        st.write(res.json()['solution'])

