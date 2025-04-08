import numpy as np
import openai
import json
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from pandasai.llm.azure_openai import AzureOpenAI
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter


#OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "57e7b0f37d0f4dc1b657496a726156c0"

pdf_path = ""


def load_document(file):
    from langchain.document_loaders import PyPDFLoader
    print(f'loading the {file}')
    loader = PyPDFLoader(file)
    bs_data = loader.load()
    return bs_data

def load_split_pdf(pdf_path):
    pdf_loader = PdfReader(open(pdf_path, "rb"))
    pdf_text = ""
    for page_num in range(len(pdf_loader.pages)):
        pdf_page = pdf_loader.pages[page_num]
        pdf_text += pdf_page.extract_text()
    #progressBar(2, 7)
    return pdf_text

def split_text_using_RCTS(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=64
    )
    split_texts = text_splitter.split_text(pdf_text)
    paragraphs = []
    for text in split_texts:
        paragraphs.extend(text.split('\n'))
    #progressBar(3, 7)
    return paragraphs

def chunk_data(bs_data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    #chunks = text_split.split_documents(bs_data)
    split_texts = text_split.split_text(bs_data)
    paragraphs = []
    for text in split_texts:
        paragraphs.extend(text.split('\n'))
    return paragraphs

def Initialize_sentence_transformer():
    sentences = ["This is an example sentence", "Each sentence is converted"]
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    #model_name = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    model = SentenceTransformer(model_name)
    #embeddings = model.encode(sentences)
    return model

Initialize_sentence_transformer()

def encode_each_paragraph(paragraphs):
    responses = []
    #model_name = "sentence-transformers/all-MiniLM-L6-v2"
    #model_name = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = Initialize_sentence_transformer()
    #model = SentenceTransformer(model_name)
    for paragraph in paragraphs:
        response = model.encode([paragraph], convert_to_tensor=True)
        responses.append((paragraph, response))
    return responses

def get_query():
    query = input("Enter your question\n")
    return query


def choose_most_relevant_sentence(responses, query):
    model = Initialize_sentence_transformer()
    query_embedding = model.encode([query], convert_to_tensor=True)
    best_response = None
    best_similarity = -1.0
    answers = []
    for paragraph, response in responses:
        # print("***", query_embedding)
        similarity = util.pytorch_cos_sim(query_embedding, response).item()

        if similarity >= 0.5:
            # count += 1
            print("yes")
            answers.append(paragraph)
    answer = "\n".join(answers)
    return answer


def ask_and_get_answer(responses, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    chat = ChatOpenAI(openai_api_key='57e7b0f37d0f4dc1b657496a726156c0', engine="sapaicx_gpt35", temperature=0.3,
                      max_tokens=1024,
                      top_p=0.95,
                      frequency_penalty=0,
                      presence_penalty=0)
    conversation = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory(
            llm=chat,

        ),
        verbose=False,
    )
    # retriever = responses.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    result = conversation.predict(input=q,context = responses)
    return result






































def calculate(path,query):

    pdf_text = load_split_pdf(path)
    paragraphs = split_text_using_RCTS(pdf_text)
    print(paragraphs[0])
    # model = Initialize_sentence_transformer
    responses = encode_each_paragraph(paragraphs=paragraphs)
    # answer = choose_most_relevant_sentence(responses=responses, query=query)
    # final_re = choose_most_relevant_sentence(responses, query)
    final_response = ask_and_get_answer(choose_most_relevant_sentence(responses, query), query)
    return final_response
    # print("The answer from model is\n", final_response)


#FastAPI Object
app = FastAPI()

# Class to pass data to post function
class DBcls(BaseModel):
    user_query: Optional[str] = ""
    file_name:Optional[str] = ""

# Welcome message from API
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

# Getting the Question response to UI
@app.post('/sapfin2')
async def answer_question(db_details: DBcls, ans: bool = True):
    print('filename$$$$$$',db_details.file_name)
    path="C:/Users/abhishek.cw.gupta/Desktop/"+db_details.file_name
    result = calculate(path, db_details.user_query)
    print('528',result)
    return result

# Class to pass data to post function
# class DBcls(BaseModel):
#     user_query: Optional[str] = ""
#     path: Optional[str] = ""
# @app.post('/sapfin2calc')
# async def answer_question(db_details: DBcls, ans: bool = True):
#     result = calculate(db_details.path, db_details.user_query)
#     print('528',result)
#     return result