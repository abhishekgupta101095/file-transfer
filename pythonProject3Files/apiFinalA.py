import re

# Note: The openai-python library support for Azure OpenAI is in preview.
import openai
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

#OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "57e7b0f37d0f4dc1b657496a726156c0"

#FastAPI Object
app = FastAPI()


# Welcome message from API
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

#Load data for UI
data = pd.read_csv("cfo_dashboard_dataset_mod.csv")
df1 = data
df1['DateMonth'] = df1['DateMonth'].astype(str)
df1['YEAR'] = df1['DateMonth'].str.slice(0, 4)
df1['MONTH'] = df1['DateMonth'].str.slice(4, 6)
# df['QUARTER'] = np.where(df['MONTH'].isin(q1m), df['Budget'] * 0.78125, df['Budget'])
df1['DT_CONV'] = pd.to_datetime(df1['DateMonth'], format='%Y%m', errors='coerce').dropna()
df1['QUARTER'] = df1['DT_CONV'].dt.quarter

#BaseModel Class to pass it to post function
class DBcls(BaseModel):
    years: Optional[list] = list(df1['YEAR'].drop_duplicates())
    col_name: Optional[list] = ['G_L_AccountDescription', 'Company_CodeDescription', 'DateMonth', 'CategoryVersion', 'Cost_CenterDescription', 'Profit_CenterDescription', 'ProductDescription', 'CustomerDescription', 'Distribution_ChannelDescription', 'Value', 'QUARTER']
    category: Optional[list] = ['Actual']
    months: Optional[list] = list(df1['MONTH'].drop_duplicates())
    quarters: Optional[list] = list(df1['QUARTER'].drop_duplicates())
    product_name: Optional[list] = list(df1['ProductDescription'].drop_duplicates())
    company_code: Optional[list] = list(df1['Company_CodeDescription'].drop_duplicates())
    profit_center_name: Optional[list] = list(df1['Profit_CenterDescription'].drop_duplicates())
    analysis: str = 'Comparative'
    user_query: Optional[str]


# Load the MPNet model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Reading Parquet file
df_user_ex = pd.read_parquet("UserQueryExample.parquet")

# Saving embeddings and solution as list
embeddings = df_user_ex['Embeddings'].tolist()
solution = df_user_ex['Solution'].tolist()

# To pass input to the model and generate response
def generate_response(payload):
    response = openai.ChatCompletion.create(
        engine="sapaicx_gpt35",
        messages=payload,
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
async def answer_question(db_details: DBcls, ans: bool = True):
    data = pd.read_csv("cfo_dashboard_dataset_mod.csv")

    # Convert into a Dataframe
    df = data
    df = df.dropna()
    df['DateMonth'] = df['DateMonth'].astype(str)
    df['YEAR'] = df['DateMonth'].str.slice(0, 4)
    df['MONTH'] = df['DateMonth'].str.slice(4, 6)
    df['DT_CONV'] = pd.to_datetime(df['DateMonth'], format='%Y%m', errors='coerce').dropna()
    df['QUARTER'] = df['DT_CONV'].dt.quarter
    df['Value'] = df.apply(lambda row: (row['Value'] * (-1) if (row['Account type'] == 'INC' or row['Account type'] == 'LEQ') else (row['Value'])), axis=1)

    # For default analysis that is Comparative
    if db_details.analysis == "Comparative":
        df = df[(df['Profit_CenterDescription'].isin(db_details.profit_center_name)) & (df['YEAR'].isin(db_details.years)) & (df["CategoryVersion"].isin(db_details.category)) & (df['MONTH'].isin(db_details.months))  & (df["ProductDescription"].isin(db_details.product_name)) & (df["Company_CodeDescription"].isin(db_details.company_code))  & (df["QUARTER"].isin(db_details.quarters))]
    # For Yearly analysis
    else:
        df = df[df['YEAR'].isin(db_details.years)]
    df = df[['G_L_AccountDescription', 'Company_CodeDescription', 'DateMonth', 'CategoryVersion', 'Cost_CenterDescription', 'Profit_CenterDescription', 'ProductDescription', 'CustomerDescription', 'Distribution_ChannelDescription', 'Value', 'QUARTER']]

    # System message to pass to the model
    system_message: str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into a variable 'df'. 
    The Database is a collection of data (Dimensions) to record the Planning and Actual numbers of various Profit centers. Project Managers and Analysts can use this data to do planning, budgeting, forecasting, and reporting.
    
    Your Task is to do the following:
        * Based on the 'Database Details' and 'user question', provide the necessary code to obtain the results as shown in the 'Example'
        * Assume you have the dataframe already loaded into a variable 'df'.
        * The User will constantly be providing you with the output to your generated code. Use that as FEEDBACK to improve.
        * In case the User provides the expected response. Generate NO FURTHER CODE
        * If your FINAL ANSWER is a DataFrame and write to 'output.csv' else write to 'output.txt'.
    
    IMPORTANT:
        * Refer to the columns present in the 'Database Details' only.
        * Make sure to provide the Generated Python code in proper formatting and Markdown.
        * ALWAYS Provide the Entire Python Code to get the solution. 
        * In Case you need some intermediate information, make sure to do that as a control flow statement within the Generated Python Code
        * NEVER print the FINAL ANSWER, if your FINAL ANSWER is a DataFrame then write to 'output.csv' else write to 'output.txt'.
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
            "column_description": "It is the path that products take from the manufacturer to the customer. Distribution channels can be direct (e.g., selling products through a company's own website) or indirect (e.g., selling products through retail stores)."
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

    system_message2: str = """
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
    
    #Database Details:
    
    [
        {
            "column_name": "CategoryVersion", 
            "data_type": "object", 
            "sample_data": ['Actual', 'Plan'], 
            "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
        },
        {
            "column_name": "'DateMonth'", 
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
            "column_description": "It is the path that products take from the manufacturer to the customer. Distribution channels can be direct (e.g., selling products through a company's own website) or indirect (e.g., selling products through retail stores)."
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
    top_results = torch.topk(cosine_scores, k=2)

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
                    final_answer = final_answer.replace([np.inf, -np.inf, np.nan], 0)
                    # To convert to string format for Fastapi
                    if 'QUARTER' in final_answer.columns:
                        final_answer['QUARTER']=final_answer['QUARTER'].values.astype(str)
                    if 'DateMonth' in final_answer.columns:
                        final_answer['DateMonth'] = final_answer['DateMonth'].values.astype(str)
                        final_answer['DateMonth'] = final_answer['DateMonth'].apply(lambda x: f"{x[:4]}/{x[4:]}")
                    final_answer.columns = final_answer.columns.str.replace('Description','')


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
@app.post('/gene')
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