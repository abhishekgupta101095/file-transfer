
def stream():
    print('first line')
    import streamlit as st
    print('after streamlit import')
    from streamlit_chat import message
    import requests
    import json
    # res = requests.request(method='post',url="http://127.0.0.1:8092/sapfin")
    print('single')
    user_q = st.chat_input()
    a=st.chat_message('human')
    a.write(user_q)
    i=st.chat_message('assistant')

