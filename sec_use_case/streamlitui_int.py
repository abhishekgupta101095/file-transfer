import streamlit as st
from streamlit_chat import message
import requests
import json

st.title("Intelligent Data Collection bot")
st.sidebar.title("Solution Panel")
with st.sidebar:
    file=st.file_uploader('upload the File:', type='pdf')
    # if file:

        # print('filename',file)
    # st.button('Read the content')

user_query_input = st.chat_input('Enter the queries')

if user_query_input:
    message((user_query_input),is_user=True)
    # st.write("Filename: ", file.name)
    inputs = {"user_query": user_query_input,'file_name':file.name}
    res = requests.post(url = "http://127.0.0.1:8092/sapfin2",json=inputs)
    print('res',res)
    print('res.text',res.text)
    # data = json.dumps(inputs))
    st.write(res.text)
    # st.write(res.json())
    # st.sidebar.write(res.json()['solution'])
    # st.write(res.json()['final_answer'])