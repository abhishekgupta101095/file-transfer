import re

# Note: The openai-python library support for Azure OpenAI is in preview.
import pdfplumber
import pandas as pd
import numpy as np
import openai
import json
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, date
import streamlit as st

from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from pandasai.llm.azure_openai import AzureOpenAI

#OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "57e7b0f37d0f4dc1b657496a726156c0"

#FastAPI Object
app = FastAPI()

# Class to pass data to post function
class DBcls(BaseModel):
    user_query: Optional[str] = ""
    file_name: Optional[str] = ""


# Welcome message from API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Collection API"}

@app.post("/process_file")
async def process_file():
      # Process the file
    processed_df = process_file("worklist.csv")  # Replace "your_file.csv" with your file path

    # Convert DataFrame to JSON
    processed_json = processed_df.to_json(orient="records")

    # Return the processed DataFrame as JSON
    return JSONResponse(content=processed_json)

data = pd.ExcelFile(r"C:\Users\abhishek.cw.gupta\PycharmProjects\pythonProject4\Gen AI - Collection Use case data_V1.xlsx")

# Getting the data for sheet FBL5N Table
data_fbl5n = data.parse('FBL5N Report ')
data_fbl5n = data_fbl5n[['BP ','Customer','Clearing Document ','Net Due Date','Document Date', 'Document Type', 'Document Number','Payment Date', 'Arrears by net due date', 'Amount in Local Currency']]
data_fbl5n['Business Partner'] = data_fbl5n['BP ']
data_fbl5n = data_fbl5n.drop('BP ',axis=1)
data_fbl5n['Clearing Document'] = data_fbl5n['Clearing Document ']
data_fbl5n=data_fbl5n.drop('Clearing Document ',axis=1)
data_fbl5n['Document Date']= pd.to_datetime(data_fbl5n['Document Date'])

# Reading the data
data_knc = data.parse('KNC1')
data_knc = data_knc[['Customer', 'Company Code', 'Fiscal Year',
                     'Balance Carryforward', 'Debit', 'Credit', 'Sales', 'Clearing Amount']]

# Calculating DSO(Days of sales outstanding)
data_knc['Balance Carryforward'] = data_knc['Balance Carryforward'].astype('str')
data_knc['Balance Carryforward'] = data_knc['Balance Carryforward'].str.replace(',', '')
data_knc['Balance Carryforward'] = data_knc['Balance Carryforward'].astype('float')
data_knc['Sales'] = data_knc['Sales'].astype('str')
data_knc['Sales'] = data_knc['Sales'].str.replace(',', '')
data_knc['Sales'] = data_knc['Sales'].astype('float')
data_knc['Business Partner'] = data_knc['Customer']
data_knc=data_knc.drop('Customer',axis=1)
data_knc.loc[data_knc['Business Partner']==18,'Customer'] = 'Amazon.com'
data_knc.loc[data_knc['Business Partner']==21,'Customer'] = 'Walmart Inc.'
data_knc.loc[data_knc['Business Partner']==32,'Customer'] = 'Target Corp'



# Get today's date
data_fbl5n['Present Date'] = date.today()

data_fbl5n['Amount in Local Currency'] = [float(str(i).replace(",", "")) for i in data_fbl5n['Amount in Local Currency']]

data_fbl5n['Arrears by net due date']=data_fbl5n['Arrears by net due date'].replace(np.nan,0)

# Reading the data from Disputed Items data
data_dis = data.parse('Disputed Items')
data_dis = data_dis[['Dispute Case ID', 'Business Partner', 'Invoice Document', 'Disputed Amount', 'Status of Dispute', 'Reason of Dispute', 'No of Days Dispute is Open']]
data_dis.loc[data_dis['Business Partner']==18,'Customer'] = 'Amazon.com'
data_dis.loc[data_dis['Business Partner']==21,'Customer'] = 'Walmart Inc.'
data_dis.loc[data_dis['Business Partner']==32,'Customer'] = 'Target Corp'

data_worklist = data.parse('Collection Worklist Mf')
data_worklist['Customer'] = data_worklist['Name of Business Partner']
data_worklist = data_worklist.drop('Name of Business Partner',axis=1)
data_worklist = data_worklist[['Business Partner','Outstanding']]

df_fbl5n = data_fbl5n

df_knc = data_knc

df_dis = data_dis

df_wl = data_worklist

# Load the MPNet model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Reading Parquet file
df_user_ex = pd.read_parquet(r"C:\Users\abhishek.cw.gupta\PycharmProjects\pythonProject4\UserQueryExample.parquet")

# Saving embeddings and solution as list
embeddings = df_user_ex['Embeddings'].tolist()
solution = df_user_ex['Solution'].tolist()

# To pass input to the model and generate response
def generate_response(payload):
    memory = ConversationBufferWindowMemory(k=1)
    chat = ChatOpenAI(openai_api_key='57e7b0f37d0f4dc1b657496a726156c0', engine="sapaicx_gpt35", temperature=0.3,
        max_tokens=1024, top_p=0.95, frequency_penalty=0, presence_penalty=0)
    conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferWindowMemory(
        llm=chat,   
        ),
    verbose=False,
    )
    print('inside gen payload',payload,'INSIDE gEN PAYLOAD END')
    response = conversation.predict(input=str(payload))
    #response = conversation.run((content=str(input=payload)))
    '''
    response = openai.ChatCompletion.create(
        engine="sapaicx_gpt4",
        messages=payload,
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    '''
    print('############ above response')
    print('*******########Inside gen response',response)
    print('################# below response')
    return response



# Getting the Question response to UI
@app.post('/intelligent_collection')
async def answer_question(db_details: DBcls, ans: bool = True):

    # System message to pass to the model
    system_message: str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into variables 'df_fbl5n' and 'df_knc'. 
    The Database is a collection of data (Dimensions) to record the invoice Payment strategies of the customer. Managers and Analysts can use this data to do planning, understand the default customers, if the customer pays the invoices in time and reporting.
    
    Your Task is to do the following:
        * Based on the 'Database Details' and 'user question', provide the necessary code to obtain the results as shown in the 'Example'
        * Assume you have the dataframe already loaded into variables 'df_fbl5n' and 'df_knc'.
        * The User will constantly be providing you with the output to your generated code. Use that as FEEDBACK to improve.
        * In case the User provides the expected response. Generate NO FURTHER CODE
        * If your FINAL ANSWER is a DataFrame and write to 'output.csv' else write to 'output.txt'.
    
    IMPORTANT:
        * Refer to the columns present in the 'Database Details' only.
        * Make sure to provide the Generated Python code in proper formatting and Markdown.
        * ALWAYS Provide the Entire Python Code to get the solution. 
        * In Case you need some intermediate information, make sure to do that as a control flow statement within the Generated Python Code
        * NEVER print the FINAL ANSWER, if your FINAL ANSWER is a DataFrame then write to 'output.csv' else write to 'output.txt'.
    #Database Details for 'df_fbl5n':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Document Type", 
            "data_type": "object", 
            "sample_data":['RV', 'DA'], 
            "column_description": "Document type is a classfication of accounting documents 'RV', 'DA'."
        },
        {
            "column_name": "Net Due Date", 
            "data_type": "datetime64[ns]", 
            "sample_data": [2023-12-21, 2023-10-02], 
            "column_description": "It refers to the outstanding or overdue payments that have not been settled by their respective due date. Negative gives the amount payment is done in advance. Positive gives a delayed payments"
        },
        {
            "column_name": "Amount in Local Currency", 
            "data_type": "float64", 
            "sample_data": [1520.12, 691.12,-636, 3040.25], 
            "column_description": "It refers to the monetary value of a transaction in the primary currency used for reporting within a specific company code."
        },
        {
            "column_name": "Document Date", 
            "data_type": "datetime.datetime", 
            "sample_data": ['2022-12-15T00:00:00.000000000', '2023-03-27T00:00:00.000000000', '2023-05-18T00:00:00.000000000', '2023-05-20T00:00:00.000000000'], 
            "column_description": "It refers to the date of payment document when submitted."
        },
        {
            "column_name": "Document Number", 
            "data_type": "numpy.int64", 
            "sample_data": [9300000001, 9300000002, 9300000003, 9300000004, 9300000005], 
            "column_description": "It refers to the document number of payment document."
        },
        {
            "column_name": "Payment Date", 
            "data_type": "datetime.datetime", 
            "sample_data": [datetime.datetime(2023, 1, 4, 0, 0), datetime.datetime(2023, 4, 16, 0, 0), datetime.datetime(2023, 5, 23, 0, 0), datetime.datetime(2023, 5, 25, 0, 0)], 
            "column_description": "It refers to the date in which payment is made."
        },
        {
            "column_name": "Arrears by net due date", 
            "data_type": "numpy.float64", 
            "sample_data": [ nan,  -2.,   3.,   7.,   6.,  22.,  23.,  98.,  16.,  26., -68., 9.], 
            "column_description": "It refers to the arrears at the time of net due date."
        },
        {
            "column_name": "Clearing Document", 
            "data_type": "numpy.float64", 
            "sample_data": [           nan, 2.00010525e+09, 2.00010708e+09, 2.00010876e+09], 
            "column_description": "It refers to the document to clear all the dues."
        }
        ]
    
    #Databases Details for 'df_knc':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Company Code", 
            "data_type": "numpy,int64", 
            "sample_data": ['9001'], 
            "column_description": "The unique code of the company."
        },
        {
            "column_name": "Fiscal Year	", 
            "data_type": "numpy.int64", 
            "sample_data": ['2004'], 
            "column_description": "Fiscal Year represents the current Fiscal year."
        },
        {
            "column_name": "Balance Carryforward", 
            "data_type": "numpy.float64", 
            "sample_data": [36550000.,  2352732., 45600310.], 
            "column_description": "This represents the carryforward balance of last fiscal year."
        },
        {
            "column_name": "Debit", 
            "data_type": "numpy.int64", 
            "sample_data": [ 17800000,   3000000, 108680000], 
            "column_description": "This represents the debit balance of the fiscal year."
        },
        {
            "column_name": "Credit", 
            "data_type": "numpy.int64", 
            "sample_data": [0,0,0], 
            "column_description": "This represents the credit balance of the fiscal year."
        },
        {
            "column_name": "Sales", 
            "data_type": "numpy.float64", 
            "sample_data": [ 17800000,   3000000, 108680000], 
            "column_description": "This represents the sales the fiscal year."
        },
        {
            "column_name": "Clearing Amount", 
            "data_type": "numpy.int64", 
            "sample_data": [0,0,0], 
            "column_description": "This represents the clearing amount balance of the fiscal year."
        }       
        ]
        
    #Database Details for 'df_dis':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Dispute Case ID", 
            "data_type": "object", 
            "sample_data":['DC001', 'DC002', 'DC003', 'DC004'], 
            "column_description": "This refers to the unique IDs of Disputed cases."
        },
        {
            "column_name": "Invoice Document", 
            "data_type": "numpy.int64", 
            "sample_data": [9300000009, 9300000010, 9300000016, 9300000019], 
            "column_description": "It refers to the Invoice Document which has been disputed."
        },
        {
            "column_name": "Disputed Amount", 
            "data_type": "object", 
            "sample_data": [nan, 'RV'], 
            "column_description": "It refers to the monetary value of Disputed amount."
        },
        {
            "column_name": "Status of Dispute", 
            "data_type": "object", 
            "sample_data": ['Amount to be collected ', 'Open', 'Resolved'], 
            "column_description": "It refers to the current status of the disputed case."
        },
        {
            "column_name": "Reason of Dispute", 
            "data_type": "object", 
            "sample_data": ['Invoice Issue', 'Delivery Issue', 'Bad Debt'], 
            "column_description": "It refers to the reason due to which dispute happened."
        },
        {
            "column_name": "No of Days Dispute is Open", 
            "data_type": "numpy.float64", 
            "sample_data": ['nan'], 
            "column_description": "It refers to the number of days for which the disputed case was open."
        }
        ]
        
    #Database Details for 'df_wl':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Outstanding", 
            "data_type": "numpy.int64", 
            "sample_data":[32353000000,    30020000,     7468820], 
            "column_description": "The amount which is outstanding for particular Business Partner."
        }
        ]
    """

    system_message2: str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into variables 'df_flb5n' and 'df_knc. 
    The Database is a collection of data (Dimensions) to record the Planning and Actual numbers of various Profit centers. Project Managers and Analysts can use this data to do planning, budgeting, forecasting, and reporting.
    You are provided with a conversation that has taken place.
    
    Your Task is to do the following:
        * Based on the 'Database Details' and 'user question', go through the rest of the conversation.
        * 'DEDUCE' what additional data is missing or shall be necessary to obtain the result.
        
    IMPORTANT:
        * Refer to the columns present in the 'Database Details' only.
        * Make sure to provide the Generated Answer in proper formatting and Markdown.
        * DO NOT GENERATE any CODE in your responses. Provide only the possible Issue/Additional information needed.
    
        #Database Details for 'df_fbl5n':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Document Type", 
            "data_type": "object", 
            "sample_data":['RV', 'DA'], 
            "column_description": "Document type is a classfication of accounting documents 'RV', 'DA'."
        },
        {
            "column_name": "Net Due Date", 
            "data_type": "datetime64[ns]", 
            "sample_data": [2023-12-21, 2023-10-02], 
            "column_description": "It refers to the outstanding or overdue payments that have not been settled by their respective due date. Negative gives the amount payment is done in advance. Positive gives a delayed payments"
        },
        {
            "column_name": "Amount in Local Currency", 
            "data_type": "float64", 
            "sample_data": [1520.12, 691.12,-636, 3040.25], 
            "column_description": "It refers to the monetary value of a transaction in the primary currency used for reporting within a specific company code."
        },
        {
            "column_name": "Document Date", 
            "data_type": "datetime.datetime", 
            "sample_data": ['2022-12-15T00:00:00.000000000', '2023-03-27T00:00:00.000000000', '2023-05-18T00:00:00.000000000', '2023-05-20T00:00:00.000000000'], 
            "column_description": "It refers to the date of payment document when submitted."
        },
        {
            "column_name": "Document Number", 
            "data_type": "numpy.int64", 
            "sample_data": [9300000001, 9300000002, 9300000003, 9300000004, 9300000005], 
            "column_description": "It refers to the document number of payment document."
        },
        {
            "column_name": "Payment Date", 
            "data_type": "datetime.datetime", 
            "sample_data": [datetime.datetime(2023, 1, 4, 0, 0), datetime.datetime(2023, 4, 16, 0, 0), datetime.datetime(2023, 5, 23, 0, 0), datetime.datetime(2023, 5, 25, 0, 0)], 
            "column_description": "It refers to the date in which payment is made."
        },
        {
            "column_name": "Arrears by net due date", 
            "data_type": "numpy.float64", 
            "sample_data": [ nan,  -2.,   3.,   7.,   6.,  22.,  23.,  98.,  16.,  26., -68., 9.], 
            "column_description": "It refers to the arrears at the time of net due date."
        },
        {
            "column_name": "Clearing Document", 
            "data_type": "numpy.float64", 
            "sample_data": [           nan, 2.00010525e+09, 2.00010708e+09, 2.00010876e+09], 
            "column_description": "It refers to the document to clear all the dues."
        }
        ]
    
    #Databases Details for 'df_knc':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Company Code", 
            "data_type": "numpy,int64", 
            "sample_data": ['9001'], 
            "column_description": "The unique code of the company."
        },
        {
            "column_name": "Fiscal Year	", 
            "data_type": "numpy.int64", 
            "sample_data": ['2004'], 
            "column_description": "Fiscal Year represents the current Fiscal year."
        },
        {
            "column_name": "Balance Carryforward", 
            "data_type": "numpy.float64", 
            "sample_data": [36550000.,  2352732., 45600310.], 
            "column_description": "This represents the carryforward balance of last fiscal year."
        },
        {
            "column_name": "Debit", 
            "data_type": "numpy.int64", 
            "sample_data": [ 17800000,   3000000, 108680000], 
            "column_description": "This represents the debit balance of the fiscal year."
        },
        {
            "column_name": "Credit", 
            "data_type": "numpy.int64", 
            "sample_data": [0,0,0], 
            "column_description": "This represents the credit balance of the fiscal year."
        },
        {
            "column_name": "Sales", 
            "data_type": "numpy.float64", 
            "sample_data": [ 17800000,   3000000, 108680000], 
            "column_description": "This represents the sales the fiscal year."
        },
        {
            "column_name": "Clearing Amount", 
            "data_type": "numpy.int64", 
            "sample_data": [0,0,0], 
            "column_description": "This represents the clearing amount balance of the fiscal year."
        }       
        ]
    
    #Database Details for 'df_dis':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Dispute Case ID", 
            "data_type": "object", 
            "sample_data":['DC001', 'DC002', 'DC003', 'DC004'], 
            "column_description": "This refers to the unique IDs of Disputed cases."
        },
        {
            "column_name": "Invoice Document", 
            "data_type": "numpy.int64", 
            "sample_data": [9300000009, 9300000010, 9300000016, 9300000019], 
            "column_description": "It refers to the Invoice Document which has been disputed."
        },
        {
            "column_name": "Disputed Amount", 
            "data_type": "object", 
            "sample_data": [nan, 'RV'], 
            "column_description": "It refers to the monetary value of Disputed amount."
        },
        {
            "column_name": "Status of Dispute", 
            "data_type": "object", 
            "sample_data": ['Amount to be collected ', 'Open', 'Resolved'], 
            "column_description": "It refers to the current status of the disputed case."
        },
        {
            "column_name": "Reason of Dispute", 
            "data_type": "object", 
            "sample_data": ['Invoice Issue', 'Delivery Issue', 'Bad Debt'], 
            "column_description": "It refers to the reason due to which dispute happened."
        },
        {
            "column_name": "No of Days Dispute is Open", 
            "data_type": "numpy.float64", 
            "sample_data": ['nan'], 
            "column_description": "It refers to the number of days for which the disputed case was open."
        }
        ]
        
    #Database Details for 'df_wl':
    [
         {
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
        {
            "column_name": "Outstanding", 
            "data_type": "numpy.int64", 
            "sample_data":[32353000000,    30020000,     7468820], 
            "column_description": "The amount which is outstanding for particular Business Partner."
        }
        ]

    """

# Code0, in case we have to provide any additional code previous to the Model provided code.
    code0: str = """
#
"""

    # User Query
    # db_details.user_query = f"""
    #            Answer the following question: {db_details.user_query}"""

    # Embedding to check with the already loaded Parquet file
    query_embedding = model.encode(db_details.user_query)

    # Compute the cosine similarity between the query and the corpus
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Sort the scores in descending order
    cosine_scores = cosine_scores.cpu()

    # Set value of k for number of examples you want
    top_results = torch.topk(cosine_scores, k=1)

    # Update the system message as follows:
    i = 0
    top_2_solutions = ''
    for indic in top_results[1]:
        top_2_solutions = top_2_solutions + solution[int(indic)]
        i = i + 1
    system_message = system_message + top_2_solutions

    # Set value of k for number of examples you want (in case tokken limit exceeds)
    top_results = torch.topk(cosine_scores, k=1)

    # Update the system message as follows:
    i = 0
    top_1_solution = ''
    for indic in top_results[1]:
        top_1_solution = top_1_solution + solution[int(indic)]
        i = i + 1
    system_message_ex = system_message + top_1_solution

    # Payload will be passed in the model
    payload = [{"role": "system", "content": system_message}, {"role": "user", "content": db_details.user_query}]
    payload_ex = [{"role": "system", "content": system_message_ex}, {"role": "user", "content": db_details.user_query}]

    # System message in case it is unable to generate the correct code
    reboot_message = """
                    SYSTEM MESSAGE: Sorry I am still learning and might go off course sometimes. 
                    Seems like you are trying to reference a data that might not be present in the DataFrame.
                    Could you please rephrase your question or refer to the SOLUTION PANEL for more details.
                    """

    # for Number of iterations
    while 1:
        counter = len(payload) / 2
        try:
            if counter < 5:
                print("Iteration " + str(counter))
                try:
                    output = "Sorry I am still learning and might go off course sometimes"
                    output = generate_response(payload)
                except:
                    payload = payload_ex
                    output = generate_response(payload)
                # print(output)
                try:
                    matches = re.findall(r"```([\s\S]*?)```", output)
                    code = "#" + " \n\n".join(matches)
                except:
                    code = "#"

                # Code execution
                exec(code0 + code)
                try:
                    final_answer = pd.read_csv('output.csv')
                    print(final_answer.info())
                    for i in final_answer.columns:
                        print("INTO the LOOP*********")
                        if isinstance((final_answer[i][0]), (np.float64, np.int64, np.bool_)) :
                           print("####TRUE64")
                           final_answer[i]=final_answer[i].values.astype(str) 
                    final_answer = final_answer.replace([np.inf, -np.inf, np.nan], 0)                                  
                    # To convert to string format for Fastapi
                    '''
                    if 'QUARTER' in final_answer.columns:
                        final_answer['QUARTER']=final_answer['QUARTER'].values.astype(str)               
                    if 'DateMonth' in final_answer.columns:
                        final_answer['DateMonth'] = final_answer['DateMonth'].values.astype(str)
                        final_answer['DateMonth'] = final_answer['DateMonth'].apply(lambda x: f"{x[:4]}/{x[4:]}")
                    final_answer.columns = final_answer.columns.str.replace('Description','')
                    final_answer.columns = final_answer.columns.str.replace('Value','Amount $')
                    for i in final_answer.columns:
                        if isinstance((final_answer[i][0]), np.float64) :
                            if (abs(final_answer[i].min())) > 1000000:
                                    final_answer[i] = round((final_answer[i] / 1000000), 2)
                                    final_answer.rename(columns={i: str(i) + '(in millions $)'}, inplace=True)
                    '''
                    #final_answer = final_answer.to_json()
                    # Creating an Output dataframe
                    df = pd.DataFrame({})
                    df.to_csv('output.csv', index=False)

                # If output is in text format
                except:
                    with open('output.txt') as f:
                        final_answer = f.read()
                    with open('output.txt', 'w') as f:
                        f.write("")
                return {'solution': output, 'final_answer': final_answer}
            else:
                output = generate_response([{"role": "system", "content": system_message2},
                                            {"role": "user", "content": json.dumps(payload[1:])}])
                return {'solution': output, 'final_answer': reboot_message}

        except Exception as e:
            error_msg = "ERROR: " + str(repr(e))
            print(error_msg)
            payload.append({"role": "assistant", "content": output})
            payload.append({"role": "user", "content": error_msg})

# Class to pass data to post function
class Explain(BaseModel):
    ques: Optional[str] = "profit"

# Getting the financial terms to UI
@app.post('/collection_strategy')
async def generate_response_gft(ques_cls: Explain):

    #
    system_message3 = """
    Assume you are a Financial Expert and you have very deep understanding of Financial terms.
    
    Important Instructions:
    * Answer should be in Financial language.
    * Answer should be accurate and precise.
    * Answer in minimum words possible.
    * Only provide factual information.
    * DON'T hallucinate the response.
    """

    #
    messages = [{'role': 'system', 'content': system_message3},
                {'role': 'user', 'content': ques_cls.ques}]
    response = openai.ChatCompletion.create(
        engine="sapaicx_gpt35",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

'''
# Getting the date values to UI
@app.get('/date_values')
async def retrieve_date():
    # Read the csv file
    data = pd.read_csv("cfo_dashboard_dataset_mod.csv")

    df = data
    df = df.dropna()
    df['DT_CONV'] = pd.to_datetime(df['DateMonth'], format='%Y%m', errors='coerce').dropna()
    date_values = df['DT_CONV'].dt.year.unique()
    return {"date_values": date_values.tolist()}          

# Getting the KPI values to the UI
@app.get('/kpi_values')
async def retrieve_kpis():
    # Read the csv file
    data = pd.read_csv("cfo_dashboard_dataset_mod.csv")

    df = data
    df = df.dropna()
    columns = [ 'Profit_CenterDescription',
                'ProductDescription',
                'G_L_AccountDescription']

    df = df[columns]
     #A dictionary to map laymen terms to actual column name
    dic = {'Profit_Center': 'Profit_CenterDescription',
            'Product': 'ProductDescription',
            'GL Account': 'G_L_AccountDescription'}
    
    kpis_ui = list(dic.keys())
    return {"date_values": kpis_ui}

'''