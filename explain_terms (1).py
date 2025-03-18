#import streamlit as st
#from streamlit_chat import message
from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
import openai
from typing import Optional


openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "57e7b0f37d0f4dc1b657496a726156c0"
prompt = ' Tell me the defention of profit margin'

app = FastAPI()

# roles ==> system user, assisstant
@app.get('/')
def read_root():
    return {"message": "Welcome from the API"}

class Explain(BaseModel):
  ques: Optional[str] = "profit"

#exp = Explain()
systemMessage ="Assume a dataanalyst looking at the financial data"

@app.post('/gene')
async def generate_response(ques_cls:Explain):
    messages = [{'role': 'system', 'content' : 'Answer as concisely as possible'},
           {'role': 'user', 'content': ques_cls.ques}]
    response = openai.ChatCompletion.create(
        engine="sapaicx_gpt35",
        messages = messages,
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']



#def app1():
     
    # Display the selected database
    # st.title("Finance Related Questions")


    # Create a chat input widget to get user questions
    #user_query = st.chat_input("Ask a Finance question")

    # If the user enters a question, send it to the answer_question function
    #if user_query:
        # Display the user question as a chat message
        #message(user_query, is_user=True)

        # Call the answer_question function and capture its output
        #output = generate_response(user_query)

        # Display the output as a chat message
        #st.write(output)
