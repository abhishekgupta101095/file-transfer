import os
import openai
import re
import pprint

openai.api_type = "azure"
openai.api_base = "https://abcgenaidemo.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "1b5b9b1029c4421d847bf494fbf7d4ec"

def update_messages(message_text, role, content):
  message_text.append({'role':role,'content':content})
  return message_text

def generate_res(message_text):
  completion = openai.ChatCompletion.create(
    engine="GPT35TURBO",
    messages=message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
  )
  print('content is---', completion['choices'][0]['message']['content'], '---')
  return completion['choices'][0]['message']['content']

message_text = [{"role":"system","content":"You are an AI assistant that provides python code only to solve complex questions. * Ask for the required information from user if data provided by user is not sufficient."}]
# user_dict = {"role":"user","content":''}
while True:
  # pprint.pprint(message_text)
  user_query = input('Enter query:')
  # user_dict['content'] = user_query
  # message_text.append(user_dict)
  message_text = update_messages(message_text,'user',user_query)
  completion=generate_res(message_text)
  matches = re.findall(r"```([\s\S]*?)```", completion)
  if matches:
    code = "#" + " \n\n".join(matches)
    exec(code)
    message_text = [{"role": "system", "content": "You are an AI assistant that provides python code only to solve complex questions. * Ask for the required information from user if data provided by user is not sufficient."}]
    # user_dict = {"role": "user", "content": ''}
    # pprint.pprint(message_text)
    # print('user_dict now is',user_dict)
    break
  else:
    message_text = update_messages(message_text,'assistant',completion)
    print('message_text',message_text)

# try:
#   x=4
# except SyntaxError:
#   print('it is syntax error')
