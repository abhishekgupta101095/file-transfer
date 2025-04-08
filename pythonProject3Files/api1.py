# import streamlit as st
# from streamlit_chat import message
import re
#Note: The openai-python library support for Azure OpenAI is in preview.
import openai
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import pickle
import base64
import uvicorn

openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "57e7b0f37d0f4dc1b657496a726156c0"

app = FastAPI()
# Welcome message from API

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

class DBcls(BaseModel):
    years: Optional[list] = ["2022","2023"]
    col_name: Optional[list] = ["Version___Description","Date___FISCAL_CALPERIOD","FLOW_TYPE___Description","SAP_ALL_COMPANY_CODE___Description","SAP_FI_IFP_GLACCOUNT___Description","SAP_ALL_PROFITCENTER___Description","SAP_ALL_FUNCTIONALAREA___Description","SAP_ALL_TRADINGPARTNER___Description","AMOUNT","QUARTER"]
    category: Optional[str] = 'Actual'
    months: Optional[list] = ['01','02','03','04','05','06','07','08','09','10','11','12']
    quarters: Optional[list] = ['1','2','3','4']
    product_name: Optional[list] = ['Consulting Unit A','Product A','Product B','Shared Services','Trading Goods','Unassigned']
    company_code: Optional[list] = ['US']
    analysis: str = 'Comparative'
    user_query: Optional[str] 

# Load the MPNet model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

df_user_ex = pd.read_parquet("UserQueryExample.parquet")

#corpus = df_user_ex['User Query'].tolist()
embeddings = df_user_ex['Embeddings'].tolist()
solution = df_user_ex['Solution'].tolist()

def generate_response(payload):
    response = openai.ChatCompletion.create(
        engine="sapaicx_gpt35",
        messages = payload,
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']


# Getting the Question response to UI
@app.post('/sapfin')
async def answer_question(db_details: DBcls, ans:bool=True):
    with open('raw_data_mod.json') as f:
        # Load the json object as a dictionary
        data = json.load(f)

    df = pd.DataFrame(data["value"])
    df = df.dropna()
    df['YEAR'] = df['Date___FISCAL_CALPERIOD'].str.slice(0, 4)
    df['MONTH'] = df['Date___FISCAL_CALPERIOD'].str.slice(4,6)
    df['DT_CONV'] = pd.to_datetime(df['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
    df['QUARTER'] = df['DT_CONV'].dt.quarter



    # pickled = pickle.dumps(df)
    # pickled_b64 = base64.b64encode(pickled)
    # pickled_b64.decode('utf-8')
    # return {"data": pickled_b64.decode('utf-8')}
    system_message: str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into a variable 'df'. 
    The Database is a collection of data (Dimensions) to record the Planning and Actual numbers of various Profit centers. Project Managers and Analysts can use this data to do planning, budgeting, forecasting, and reporting.

    Your Task is to do the following:
        * Based on the 'Database Details' and 'user question', provide the necessary code to obtain the results as shown in the 'Example'
        * Assume you have the dataframe already loaded into a variable 'df'.
        * The User will constantly be providing you with the output to your generated code. Use that as FEEDBACK to improve.
        * In case the User provides the expected response. Generate NO FURTHER CODE
        * If your your FINAL ANSWER is a DataFrame and write to 'output.csv' else write to 'output.txt'.

    IMPORTANT:
        * Refer to the columns present in the 'Database Details' only.
        * Make sure to provide the Generated Python code in proper formatting and Markdown.
        * ALWAYS Provide the Entire Python Code to get the solution. 
        * In Case you need some intermediate information, make sure to do that as a control flow statement within the Generated Python Code
        * Every year is divided into 4 quarters. January - March is Quarter 1, April to June is Quarter 2 and so on.
        * Do not use 'SAP_ALL_FUNCTIONALAREA___Description' column for calculations if the data is already present in 'SAP_FI_IFP_GLACCOUNT___Description' column (For eg:- profit margin etc..).
        * The formula for Net Revenue = Gross Revenue - Sales Deduction, and ALWAYS calculate net revenue based on this formula only.
        
    #Database Details:

[
    {
        "column_name": "CategoryVersion", 
        "data_type": "object", 
        "sample_data": ['Actual', 'Plan'], 
        "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
    },
    {
        "column_name": "DateMonth", 
        "data_type": "object", 
        "sample_data": ['202201', '202202', '202203'], 
        "column_description": "The date of the data that is being analyzed. This can be used to filter data or to analyze trends over time. For example, you might want to see how sales have changed over the past year."
    },
    {
        "column_name": "Cost_CenterDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned', 'NA/PL6/HR', 'NL/PL2/HR', 'NL/PL1/HR', 'NL/PL6/HR', 'NA/PL1/HR', 'NA/PL2/HR'], 
        "column_description": "It represents a location where costs occur. Cost centers are used to collect and allocate overhead costs, and to track the costs of business activities. Cost centers can be defined by function, department, location, or any other criteria that is meaningful for the organization."
    },
    {
        "column_name": "Company_CodeDescription", 
        "data_type": "object", 
        "sample_data": ['US', 'EU'], 
        "column_description": "The company code can be used to track data for different business units or subsidiaries. For example, you might want to see how much money has been made by each of your company's subsidiaries such as Germany, India, US etc."
    },
    {
        "column_name": "G_L_AccountDescription", 
        "data_type": "object", 
        "sample_data":['Opening Cash', 'Equity Shares Outstanding', 'Interest Expense ', 'Hub Expenses ', 'Market Capitalization', 'Non Op Expenses ', 'Market Price per share', 'Operations', 'Payroll Tax & Fringe ', 'Standard Cost ', 'Investments', 'MB-4001--BOA01-USD01 ', 'Depreciation Expense ', 'Financing', 'Customer Rcvbls ', 'Inventory ', 'FA Build & Lease Imp ', 'Dividend per Share', 'AD Build & Lease Imp ', 'Sponsorships ', 'LT Investments ', 'AA Patents ', 'Vendor Payables - Do ', 'Cur Inc Tax-Natl ', 'AL Payroll ', 'Inc Tax Pay-ST/Prov ',  'Other Current Liab ', 'LT of LT Conv Debt ', 'Ord Shares / Comm St ', 'CY Ret Earnings '], 
        "column_description": "The general ledger account that is being analyzed. This can be used to track financial transactions or to analyze costs and expenses. For example, you might want to see how much money has been spent on sales in a particular month."
    },
    {
        "column_name": "Profit_CenterDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned', 'Business market Canada', 'Business market Brazil', 'Business market Argentina', 'Business market Australia'], 
        "column_description": "It can be used to track data for different product lines or customer segments. For example, you might want to see how much profit has been made by each of your product lines."
    },
    {
        "column_name": "CustomerDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": "It is an organization or individual that purchases goods or services from a company. Customers are created and managed in the SAP Customer Relationship Management (CRM) module."
    }
    {
        "column_name": "ProductDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": "It is a good or service that a company sells to its customers. Products are created and managed in the SAP Materials Management (MM) module."
    },
    {
        "column_name": "Distribution_ChannelDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": ""
    },
    {
        "column_name": "Value", 
        "data_type": "float64", 
        "sample_data": [-11000.0, 181400.0, 14150.0, -3300.0], 
        "column_description": "represents the transactional data Value"
    },
    {
        "column_name": "QUARTER", 
        "data_type": "object", 
        "sample_data": [1, 2, 3, 4],
        "column_description": "represents the quarter of the particular year"
    }
]
    """

    system_message2 : str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into a variable 'df'. 
    The Database is a collection of data (Dimensions) to record the Planning and Actual numbers of various Profit centers. Project Managers and Analysts can use this data to do planning, budgeting, forecasting, and reporting.
    You are provided with a conversation that has taken place.

    Your Task is to do the following:
        * Based on the 'Database Details' and 'user question', go through the rest of the conversation.
        * 'DEDUCE' what additional data is missing or shall be necessary to obtain the result.

    IMPORTANT:
        * Refer to the columns present in the 'Database Details' only.
        * Make sure to provide the Generated Answer in proper formatting and Markdown.
        * DO NOT GENERATE any CODE in your responses. Provide only the possible Issue/Additional information needed.
        * Do not use 'SAP_ALL_FUNCTIONALAREA___Description' column for calculations if the data is already present in 'SAP_FI_IFP_GLACCOUNT___Description' column (For eg:- profit margin etc..).

    #Database Details:

[
    {
        "column_name": "CategoryVersion", 
        "data_type": "object", 
        "sample_data": ['Actual', 'Plan'], 
        "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
    },
    {
        "column_name": "DateMonth", 
        "data_type": "object", 
        "sample_data": ['202201', '202202', '202203'], 
        "column_description": "The date of the data that is being analyzed. This can be used to filter data or to analyze trends over time. For example, you might want to see how sales have changed over the past year."
    },
    {
        "column_name": "Cost_CenterDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned', 'NA/PL6/HR', 'NL/PL2/HR', 'NL/PL1/HR', 'NL/PL6/HR', 'NA/PL1/HR', 'NA/PL2/HR'], 
        "column_description": "It represents a location where costs occur. Cost centers are used to collect and allocate overhead costs, and to track the costs of business activities. Cost centers can be defined by function, department, location, or any other criteria that is meaningful for the organization."
    },
    {
        "column_name": "Company_CodeDescription", 
        "data_type": "object", 
        "sample_data": ['US', 'EU'], 
        "column_description": "The company code can be used to track data for different business units or subsidiaries. For example, you might want to see how much money has been made by each of your company's subsidiaries such as Germany, India, US etc."
    },
    {
        "column_name": "G_L_AccountDescription", 
        "data_type": "object", 
        "sample_data":['Opening Cash', 'Equity Shares Outstanding', 'Interest Expense ', 'Hub Expenses ', 'Market Capitalization', 'Non Op Expenses ', 'Market Price per share', 'Operations', 'Payroll Tax & Fringe ', 'Standard Cost ', 'Investments', 'MB-4001--BOA01-USD01 ', 'Depreciation Expense ', 'Financing', 'Customer Rcvbls ', 'Inventory ', 'FA Build & Lease Imp ', 'Dividend per Share', 'AD Build & Lease Imp ', 'Sponsorships ', 'LT Investments ', 'AA Patents ', 'Vendor Payables - Do ', 'Cur Inc Tax-Natl ', 'AL Payroll ', 'Inc Tax Pay-ST/Prov ',  'Other Current Liab ', 'LT of LT Conv Debt ', 'Ord Shares / Comm St ', 'CY Ret Earnings '], 
        "column_description": "The general ledger account that is being analyzed. This can be used to track financial transactions or to analyze costs and expenses. For example, you might want to see how much money has been spent on sales in a particular month."
    },
    {
        "column_name": "Profit_CenterDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned', 'Business market Canada', 'Business market Brazil', 'Business market Argentina', 'Business market Australia'], 
        "column_description": "It can be used to track data for different product lines or customer segments. For example, you might want to see how much profit has been made by each of your product lines."
    },
    {
        "column_name": "CustomerDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": "It is an organization or individual that purchases goods or services from a company. Customers are created and managed in the SAP Customer Relationship Management (CRM) module."
    }
    {
        "column_name": "ProductDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": "It is a good or service that a company sells to its customers. Products are created and managed in the SAP Materials Management (MM) module."
    },
    {
        "column_name": "Distribution_ChannelDescription", 
        "data_type": "object", 
        "sample_data": ['Unassigned'], 
        "column_description": ""
    },
    {
        "column_name": "Value", 
        "data_type": "float64", 
        "sample_data": [-11000.0, 181400.0, 14150.0, -3300.0], 
        "column_description": "represents the transactional data Value"
    },
    {
        "column_name": "QUARTER", 
        "data_type": "object", 
        "sample_data": [1, 2, 3, 4],
        "column_description": "represents the quarter of the particular year"
    }
]
    """

    code0: str = """
import json
import pandas as pd

# Open the json file
with open('raw_data_mod.json') as f:
    # Load the json object as a dictionary
    data = json.load(f)

df = pd.DataFrame(data["value"])
df = df.dropna()
df['DT_CONV']  = pd.to_datetime(df['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
df['QUARTER'] = df['DT_CONV'].dt.quarter
df = df[df['FLOW_TYPE___Description']=='Closing Amount']
columns = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description', \
        'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', \
        'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',\
        'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT','QUARTER']

df = df[columns]"""
    db_details.user_query= f"""
                Answer the following question: {db_details.user_query}
                """

    if len(db_details.product_name) < len(list(df['SAP_ALL_PROFITCENTER___Description'].drop_duplicates())):
        db_details.user_query = db_details.user_query + f"""
                        For the 'SAP_ALL_PROFITCENTER___Description': {db_details.product_name}
                    """
    if len(db_details.company_code) < len(list(df['SAP_ALL_COMPANY_CODE___Description'].drop_duplicates())):
        db_details.user_query = db_details.user_query + f"""
                        For the 'SAP_ALL_COMPANY_CODE___Description': {db_details.company_code}
                    """
    if len(db_details.quarters) < len(list(df['QUARTER'].drop_duplicates())):
        db_details.user_query = db_details.user_query + f"""
                        For the 'QAUARTER' : {db_details.quarters}
                    """
    if len(db_details.months) < len(list(df['MONTH'].drop_duplicates())):
        db_details.user_query = db_details.user_query + f"""
                        For the 'MONTH'  :  {db_details.months}
                    """
    if len(db_details.years) < len(list(df['YEAR'].drop_duplicates())):
        db_details.user_query = db_details.user_query + f"""
                        For the 'YEAR' : {db_details.years}
                    """
    if len(db_details.col_name) < 10:
        db_details.user_query = db_details.user_query + f"""
                        For the 'Column' : {db_details.col_name}
                    """

    #system_message=db_details.system_message
    #system_message2=db_details.system_message2 
    #user_query=db_details.user_query="profit" 
    #code0=db_details.code0 
     # If the user enters a question, send it to the answer_question function
    
    
    # As of now we are allowing 5 iterations
    #db_details_body = await db_details.json()
    # Encode the query with the model
    #query_embedding = model.encode(db_details_body['user_query'])
    query_embedding = model.encode(db_details.user_query)
    print(db_details.user_query)

    # Compute the cosine similarity between the query and the corpus
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Sort the scores in descending order
    cosine_scores = cosine_scores.cpu()

    # Set value of k for number of examples you want
    top_results = torch.topk(cosine_scores, k=2)

    # Update the system message as follows:
    i = 0
    top_2_solutions = ''
    for indic in top_results[1]:
        top_2_solutions = top_2_solutions + solution[int(indic)]
        i = i + 1
    system_message = system_message + top_2_solutions
    print(system_message)
    top_results = torch.topk(cosine_scores, k=1)

    # Update the system message as follows:
    i = 0
    top_1_solution = ''
    for indic in top_results[1]:
        top_1_solution = top_1_solution + solution[int(indic)]
        i = i + 1
    system_message_ex = system_message + top_1_solution

    payload = [{"role":"system","content":system_message},{"role":"user","content":db_details.user_query}]
    payload_ex = [{"role": "system", "content": system_message_ex}, {"role": "user", "content": db_details.user_query}]
    #st.write(top_2_solutions)

    
    reboot_message = """
                    SYSTEM MESSAGE: Sorry I am still learning and might go off course sometimes. 
                    Seems like you are trying to reference a data that might not be present in the DataFrame.
                    Could you please rephrase your question or refer to the SOLUTION PANEL for more details.
                    """
    while 1:
        counter = len(payload)/2
        try:
            if counter<5:
                print("Iteration "+str(counter))
                try:
                    output = generate_response(payload)
                except:
                    payload=payload_ex
                    output = generate_response(payload)
                #print(output)
                try:
                    matches = re.findall(r"```([\s\S]*?)```", output)
                    code = "#"+" \n\n".join(matches)
                except:
                    code = "#"

                exec(code0+code)
                try:
                    final_answer = pd.read_csv('output.csv')
                    df = pd.DataFrame({})
                    df.to_csv('output.csv', index=False)
                    # csv_str = df.to_csv(index=False)
                    # output1 = generate_response([{"role": "user", "content":f'''Extract the data present in bullet points as text {csv_str}.'''}])
                    # st.write(output1)
                except:
                    with open('output.txt') as f:
                        final_answer = f.read()
                    with open('output.txt', 'w') as f:
                        f.write("")
                return {'solution': output, 'final_answer': final_answer}
            else:
                output = generate_response([{"role":"system","content":system_message2},{"role":"user","content":json.dumps(payload[1:])}])
                return {'solution': output, 'final_answer': reboot_message}
                
        except Exception as e:
            error_msg = "ERROR: "+str(repr(e))
            print(error_msg)
            payload.append({"role": "assistant", "content": output})
            payload.append({"role": "user", "content": error_msg})

class Explain(BaseModel):
  ques: Optional[str] = "profit"

# Getting the financial terms to UI
@app.post('/gene')
async def generate_response_gft(ques_cls:Explain):
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

# class DateRet(BaseModel):
#     years: Optional[list] = ["2022"]
#     col_name: Optional[list] = ['YEAR']
#     category: Optional[str] = 'Actual'
#     months: Optional[list] = ['10']
#     quarters: Optional[list] = ['4']
#     product_name: Optional[list] = ['Product A']
#     company_code: Optional[list] = ['']
#     analysis = ''
#
#
#
# # Getting the date values to UI
# @app.post('/date_values')
# async def retrieve_date(db_details:DateRet):
#     with open('raw_data_mod.json') as f:
#         # Load the json object as a dictionary
#         data = json.load(f)
#
#     df = pd.DataFrame(data["value"])
#     df = df.dr()
#     df['YEAR'] = df['Date___FISCAL_CALPERIOD'].str.slice(0, 4)
#
#     if db_details.analysis == 'Comparative':
#         df = df[df['YEAR'].isin(db_details.years) & df['SAP_ALL_PROFITCENTER___Description'].isin(db_details.product_name) & df['Version___Description']==db_details.category & df['MONTH'].isin(db_details.months) & df['QUARTER'].isin(db_details.quarters)]
#
#
#
#     if db_details.analysis != 'Comparative':
#         df = df[df['YEAR'].isin(db_details.years)]
#     print(df)
#     pickled = pickle.dumps(df)
#     pickled_b64 = base64.b64encode(pickled)
#     pickled_b64.decode('utf-8')
#     return {"data": pickled_b64.decode('utf-8')}



# async def retrieve_date1(db_details):
#     with open('raw_data_mod.json') as f:
#         # Load the json object as a dictionary
#         data = json.load(f)
#
#     df = pd.DataFrame(data["value"])
#     df = df.dr()
#     df['YEAR'] = df['Date___FISCAL_CALPERIOD'].str.slice(0, 4)
#     df = df[df['YEAR'].isin(db_details.years)]
#     retrieve_date(df)
#     return df
# filter_df=retrieve_date1(DateRet)
# print('a',filter_df)
# def datefilter(df,db_details):
#     if isinstance(db_details.year,str):
#         df=df[df['YEAR']==db_details.year]
#     else:
#         df=df[df['YEAR'].str.isin(db_details.year)]
#     return df
##Getting the KPI values to the UI
# @app.get('/kpi_values')
# async def retrieve_kpis():
#     # Open the json file
#     with open('raw_data_mod.json') as f:
#         # Load the json object as a dictionary
#         data = json.load(f)
#
#     df = pd.DataFrame(data["value"])
#     df = df.dr()
#     columns = ['Version___Description', 'FLOW_TYPE___Description', \
#             'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', \
#             'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',\
#             'SAP_ALL_TRADINGPARTNER___Description']
#
#     df = df[columns]
#      #A dictionary to map laymen terms to actual column name
#     dic = {'Version': 'Version___Description',
#             'Flow Type': 'FLOW_TYPE___Description',
#             'GL Account': 'SAP_FI_IFP_GLACCOUNT___Description',
#             'Functional Area': "SAP_ALL_FUNCTIONALAREA___Description",
#             'Trading Partner': "SAP_ALL_TRADINGPARTNER___Description"}
#
#     kpis_ui = list(dic.keys())
#     return {"date_values": kpis_ui}
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port='80')