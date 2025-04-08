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

data = pd.ExcelFile(r"C:\Users\abhishek.cw.gupta\Downloads\Gen AI - Collection Use case data_V1Latest11.xlsx")

# Getting the data for sheet FBL5N Table
data_fbl5n = data.parse('FBL5N Report ')

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

# Reading the ukmbp_cms data
data_ukmbp = data.parse('UKMBP_CMS')

# Reading the disputed items list
data_di = data.parse('Disputed Items')

# Retrieve Worklist data
data_worklist = data.parse('Collection Worklist Mf')

# comparing the date with today's date
from datetime import datetime, timedelta, date

# Get today's date
data_fbl5n['Present Date'] = date.today()

# Comparing the Net Due Date greater than Present Date
(data_fbl5n['Net Due Date']) > data_fbl5n['Present Date']

data_fbl5n['Amount in Local Currency'] = [float(str(i).replace(",", "")) for i in data_fbl5n['Amount in Local Currency']]

# Calculating the Payment Due sum > Present day and group by customer and Amaount sum
amount_sum = pd.DataFrame(data_fbl5n[data_fbl5n['Net Due Date'] > data_fbl5n['Present Date']].groupby('Customer')['Amount in Local Currency'].sum())

# Checking the Rule_2 condition
amount_sum['cond_2'] = (amount_sum['Amount in Local Currency'] > 50000) & (amount_sum['Amount in Local Currency'] < 1000000)

# Checking the Rule_2 condition and assigning the valuation points accordingly
amount_sum['valu_2'] = np.where(amount_sum['cond_2'] == True, 35, 0)

# index change to columns
amount_sum.reset_index(level=0, inplace=True)

# Ceating the customer column
data_ukmbp['Customer'] = ['Amazon.com','Target Corp', 'Walmart Inc.']

# Merging the dataframes with the rule
result = pd.merge(data_ukmbp, amount_sum, how="outer", on=["Customer", "Customer"])

# Checking the Rule_3 condition
amount_sum_30 = pd.DataFrame(data_fbl5n[data_fbl5n['Net Due Date'] > (data_fbl5n['Present Date'] - timedelta(days=30))].groupby('Customer')['Amount in Local Currency'].sum())

# Checking the Rule_3 condition
amount_sum['cond_3'] = (amount_sum['Amount in Local Currency'] > 50000) & (amount_sum['Amount in Local Currency'] < 10000000)

# Checking the Rule_3 condition and assigning the valuation points accordingly
amount_sum['valu_3'] = np.where(amount_sum['cond_3'] == True, 50, 0)

#Merging the dataframes
data_ukmbp = pd.merge(data_ukmbp, data_di, how="outer", on =['Business Partner',"Business Partner"])

#Merging the dataframes with the rule
data_ukmbp = pd.merge(data_ukmbp, data_fbl5n, how="outer", on=["Customer", "Customer"])

# Merging the dataframes with the rule
data_ukmbp = pd.merge(data_ukmbp, amount_sum, how="outer", on=["Customer", "Customer"])

# Rule_1 for risk class
risk_class = '|'.join(['C','D','E'])
data_ukmbp['Risk Class'].str.contains(risk_class)
data_ukmbp['valu_1'] = np.where(data_ukmbp['Risk Class'].str.contains(risk_class), 0, 20)
path = "C:/Users/abhishek.cw.gupta/Desktop/" + "Amazon-2023-Annual Report.pdf"
with pdfplumber.open(path) as f:
    print("*****",f.pages[5].extract_tables())

bs_infolist = f.pages[5].extract_tables()

# cahnge the 3d list to 2d list
import itertools
bs_infolist2d = list(itertools.chain.from_iterable(bs_infolist))

# Create a balance sheet dictionary
bs_infodict = {tuple(e[0]): tuple(e[1:]) for e in bs_infolist2d}

# Popping first element of a list
from operator import itemgetter
pop_list = list( map(itemgetter(0), bs_infolist2d))

# Creating a dictionary from the list
final_dict = dict(zip(pop_list, list(bs_infodict.values())))

# Creating o dataframe from the dictionary
bs_dataframe = pd.DataFrame(final_dict)

# Replacing the comas in columns
bs_dataframe['Total current assets'] = [str(i).replace(",", "") for i in bs_dataframe['Total current assets']]
bs_dataframe['Inventories'] = [str(i).replace(",", "") for i in bs_dataframe['Inventories']]
bs_dataframe['Total liabilities and stockholders’ equity'] = [str(i).replace(",", "") for i in bs_dataframe['Total liabilities and stockholders’ equity']]

# Use the `to_numeric()` function to convert the column to floats
bs_dataframe["Total current assets"] = pd.to_numeric(bs_dataframe["Total current assets"], errors='coerce')
bs_dataframe['Inventories'] = pd.to_numeric(bs_dataframe['Inventories'], errors='coerce')
bs_dataframe['Total liabilities and stockholders’ equity'] = pd.to_numeric(bs_dataframe['Total liabilities and stockholders’ equity'], errors='coerce')

#Find the bs_ratio
data_ukmbp['bp_ratio'] = 0
data_ukmbp['bp_ratio'][0] = (bs_dataframe["Total current assets"][3]-bs_dataframe['Inventories'][3])/bs_dataframe['Total liabilities and stockholders’ equity'][4]

# Fill nan with 0's
data_ukmbp = data_ukmbp.fillna(0)

# Adding the total valuation points
data_ukmbp['Value Total'] = data_ukmbp['valu_1'] + data_ukmbp['valu_2'] + data_ukmbp['valu_3']

# Assigning the strtegy as per the valuation totals
value_cls = [(data_ukmbp['Value Total'] == 20), (data_ukmbp['Value Total'] == 35), (data_ukmbp['Value Total'] == 55)]
data_ukmbp["Strategy"] = np.select(value_cls, ['GEN001','GEN001', 'GEN001'] )

# Initializing the summary column
data_ukmbp['Summary'] = ""
#  Assigning the strtegy as per the bp_ratio
value_cls = [(data_ukmbp['bp_ratio'] < 0.6), (data_ukmbp['bp_ratio'] != 0.0), (data_ukmbp['bp_ratio'] == 0.0) ]
data_ukmbp["Strategy Change1"] = np.select(value_cls, ['GEN00X', 'GEN00X', 'GEN001'])

# Adding up strategy change summary
for ind, item in enumerate(data_ukmbp['bp_ratio']):
    if item < 0.6 :
        data_ukmbp["Strategy Change1"][ind] = 'GEN00X'
        data_ukmbp['Summary'][ind] = f'Due to change in bp_ratio {item}, the strategy changed from GEN001 to GEN00X'
    else:
        data_ukmbp["Strategy Change1"][ind] = 'GEN001'

data_Worklist_mf = pd.merge(data_worklist,data_ukmbp[['Summary', "Business Partner", "Value Total"]], how="outer", on=["Business Partner"])
# st.dataframe(data_Worklist_mf.style.highlight_max(axis=0))

data_Worklist_mf.to_csv('worklist.csv', index = False)
df = pd.merge(data_worklist, data_ukmbp, how="outer", on=["Business Partner"])
df = pd.merge(df, data_knc, how='left', on=[ 'Business Partner','Customer'])
# df.columns
# df['Outstanding']

# st.dataframe(data_Worklist_mf)

# Merge the two dataframes
# df = data_fbln5.merge(data_knc, left_on='Account', right_on='Customer')

# Merge the two dataframes
# df = df.merge(data_UKM_MALUS_DSP, left_on='Account', right_on='Partner')

# Printing the Dataframe *******
# print(df['Valuation points'])

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
    print(payload)
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
    print(response)
    return response



# Getting the Question response to UI
@app.post('/intelligent_collection')
async def answer_question(db_details: DBcls, ans: bool = True):
    #Load data for UI
    #data = pd.ExcelFile("C:/Users/a.a.madhu.karanath/SAP Finance2/Intelligent_coll_test2.xlsx")
    data = pd.ExcelFile(r"C:\Users\abhishek.cw.gupta\Downloads\Gen AI - Collection Use case data_V1Latest11.xlsx")

    # Getting the data for sheet FBL5N Table
    data_fbl5n = data.parse('FBL5N Report ')

    # Reading the ukmbp_cms data
    data_ukmbp = data.parse('UKMBP_CMS')

    # Reading the disputed items list
    data_di = data.parse('Disputed Items')

    # Retrieve Worklist data
    data_worklist = data.parse('Collection Worklist Mf')

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

    # Reading the data
    data_promise = data.parse('Promise To Pay')

    # comparing the date with todays date
    from datetime import datetime, timedelta, date

    # Get today's date
    data_fbl5n['Present Date'] = date.today()

    # Comparing the Net Due Date greater than Present Date
    (data_fbl5n['Net Due Date']) > data_fbl5n['Present Date']

    data_fbl5n['Amount in Local Currency'] = [float(str(i).replace(",", "")) for i in
                                              data_fbl5n['Amount in Local Currency']]

    # Calculating the Payment Due sum > Present day and group by customer and Amaount sum
    amount_sum = pd.DataFrame(data_fbl5n[data_fbl5n['Net Due Date'] > data_fbl5n['Present Date']].groupby('Customer')[
                                  'Amount in Local Currency'].sum())

    # Checking the Rule_2 condition
    amount_sum['cond_2'] = (amount_sum['Amount in Local Currency'] > 50000) & (
                amount_sum['Amount in Local Currency'] < 1000000)

    # Checking the Rule_2 condition and assigning the valuation points accordingly
    amount_sum['valu_2'] = np.where(amount_sum['cond_2'] == True, 35, 0)

    # index change to columns
    amount_sum.reset_index(level=0, inplace=True)

    # Ceating the customer column
    data_ukmbp['Customer'] = ['Amazon.com', 'Target Corp', 'Walmart Inc.']

    # Merging the dataframes with the rule
    result = pd.merge(data_ukmbp, amount_sum, how="outer", on=["Customer", "Customer"])

    # Checking the Rule_3 condition
    amount_sum_30 = pd.DataFrame(
        data_fbl5n[data_fbl5n['Net Due Date'] > (data_fbl5n['Present Date'] - timedelta(days=30))].groupby('Customer')[
            'Amount in Local Currency'].sum())

    # Checking the Rule_3 condition
    amount_sum['cond_3'] = (amount_sum['Amount in Local Currency'] > 50000) & (
                amount_sum['Amount in Local Currency'] < 10000000)

    # Checking the Rule_3 condition and assigning the valuation points accordingly
    amount_sum['valu_3'] = np.where(amount_sum['cond_3'] == True, 50, 0)

    # Merging the dataframes
    data_ukmbp = pd.merge(data_ukmbp, data_di, how="outer", on=['Business Partner', "Business Partner"])

    # Merging the dataframes with the rule
    data_ukmbp = pd.merge(data_ukmbp, data_fbl5n, how="outer", on=["Customer", "Customer"])

    # Merging the dataframes with the rule
    data_ukmbp = pd.merge(data_ukmbp, amount_sum, how="outer", on=["Customer", "Customer"])

    # Rule_1 for risk class
    risk_class = '|'.join(['C', 'D', 'E'])
    data_ukmbp['Risk Class'].str.contains(risk_class)
    data_ukmbp['valu_1'] = np.where(data_ukmbp['Risk Class'].str.contains(risk_class), 0, 20)
    path = "C:/Users/abhishek.cw.gupta/Desktop/" + "Amazon-2023-Annual Report.pdf"
    with pdfplumber.open(path) as f:
        print("*****", f.pages[5].extract_tables())

    bs_infolist = f.pages[5].extract_tables()

    # cahnge the 3d list to 2d list
    import itertools
    bs_infolist2d = list(itertools.chain.from_iterable(bs_infolist))

    # Create a balance sheet dictionary
    bs_infodict = {tuple(e[0]): tuple(e[1:]) for e in bs_infolist2d}

    # Popping first element of a list
    from operator import itemgetter
    pop_list = list(map(itemgetter(0), bs_infolist2d))

    # Creating a dictionary from the list
    final_dict = dict(zip(pop_list, list(bs_infodict.values())))

    # Creating o dataframe from the dictionary
    bs_dataframe = pd.DataFrame(final_dict)

    # Replacing the comas in columns
    bs_dataframe['Total current assets'] = [str(i).replace(",", "") for i in bs_dataframe['Total current assets']]
    bs_dataframe['Inventories'] = [str(i).replace(",", "") for i in bs_dataframe['Inventories']]
    bs_dataframe['Total liabilities and stockholders’ equity'] = [str(i).replace(",", "") for i in bs_dataframe[
        'Total liabilities and stockholders’ equity']]

    # Use the `to_numeric()` function to convert the column to floats
    bs_dataframe["Total current assets"] = pd.to_numeric(bs_dataframe["Total current assets"], errors='coerce')
    bs_dataframe['Inventories'] = pd.to_numeric(bs_dataframe['Inventories'], errors='coerce')
    bs_dataframe['Total liabilities and stockholders’ equity'] = pd.to_numeric(
        bs_dataframe['Total liabilities and stockholders’ equity'], errors='coerce')

    # Find the bs_ratio
    data_ukmbp['bp_ratio'] = 0
    data_ukmbp['bp_ratio'][0] = (bs_dataframe["Total current assets"][3] - bs_dataframe['Inventories'][3]) / \
                                bs_dataframe['Total liabilities and stockholders’ equity'][4]

    # Fill nan with 0's
    data_ukmbp = data_ukmbp.fillna(0)

    # Adding the total valuation points
    data_ukmbp['Value Total'] = data_ukmbp['valu_1'] + data_ukmbp['valu_2'] + data_ukmbp['valu_3']

    # Assigning the strtegy as per the valuation totals
    value_cls = [(data_ukmbp['Value Total'] == 20), (data_ukmbp['Value Total'] == 35),
                 (data_ukmbp['Value Total'] == 55)]
    data_ukmbp["Strategy"] = np.select(value_cls, ['GEN001', 'GEN001', 'GEN001'])

    # Initializing the summary column
    data_ukmbp['Summary'] = ""
    #  Assigning the strtegy as per the bp_ratio
    value_cls = [(data_ukmbp['bp_ratio'] < 0.6), (data_ukmbp['bp_ratio'] != 0.0), (data_ukmbp['bp_ratio'] == 0.0)]
    data_ukmbp["Strategy Change1"] = np.select(value_cls, ['GEN00X', 'GEN00X', 'GEN001'])

    # Adding up strategy change summary
    for ind, item in enumerate(data_ukmbp['bp_ratio']):
        if item < 0.6:
            data_ukmbp["Strategy Change1"][ind] = 'GEN00X'
            data_ukmbp['Summary'][ind] = f'Due to change in bp_ratio {item}, the strategy changed from GEN001 to GEN00X'
        else:
            data_ukmbp["Strategy Change1"][ind] = 'GEN001'

    data_Worklist_mf = pd.merge(data_worklist, data_ukmbp[['Summary', "Business Partner", "Value Total"]], how="outer",
                                on=["Business Partner"])
    # st.dataframe(data_Worklist_mf.style.highlight_max(axis=0))

    # data_Worklist_mf.to_csv('worklist.csv', index = False)
    df = pd.merge(data_worklist, data_ukmbp, how="outer", on=["Business Partner"])

    # Adding Value at risk column
    df_t = df[df['Reason of Dispute'] == 'Bad Debt'][['Customer', 'Outstanding']].groupby(['Customer']).count() / df[
        ['Customer', 'Outstanding']].groupby(['Customer']).count()
    df_t['Value at Risk'] = df_t['Outstanding']
    df_t = df_t.drop(['Outstanding'], axis=1)
    df_t['Customer'] = df_t.index
    df_t['I'] = [0, 1, 2, 4]
    df_t = df_t.set_index('I')
    df = pd.merge(df, df_t, how='outer', on=["Customer", "Customer"])

    data_promise['Business Partner'] = data_promise['Business Partner ']
    data_promise = data_promise.drop(['Business Partner '], axis=1)
    df = pd.merge(df, data_promise[['Business Partner', 'Status of Promise To Pay']], how='outer',
                  on=['Business Partner', 'Business Partner'])

    data_Worklist_mf.to_csv('worklist.csv', index = False)
    df = pd.merge(data_worklist, data_ukmbp, how="outer", on=["Business Partner"])
    # df = pd.merge(df,data_knc, how='left',on=['Customer','Business Partner'])
    # data_Worklist_js = data_Worklist_mf.to_json(orient="records")
    # Merge the two dataframes
    #df = df.merge(data_UKM_MALUS_DSP, left_on='Account', right_on='Partner')
    # For default analysis that is Comparative
    #if db_details.analysis == "Comparative":
    #df = df[(df['Profit_CenterDescription'].isin(db_details.profit_center_name)) & (df['YEAR'].isin(db_details.years)) & (df["CategoryVersion"].isin(db_details.category)) & (df['MONTH'].isin(db_details.months))  & (df["ProductDescription"].isin(db_details.product_name)) & (df["Company_CodeDescription"].isin(db_details.company_code))  & (df["QUARTER"].isin(db_details.quarters))]
    # For Yearly analysis
    #else:
    #df = df[df['YEAR'].isin(db_details.years)]
    #df = df[['G_L_AccountDescription', 'Company_CodeDescription', 'DateMonth', 'CategoryVersion', 'Cost_CenterDescription', 'Profit_CenterDescription', 'ProductDescription', 'CustomerDescription', 'Distribution_ChannelDescription', 'Value', 'QUARTER']]

    # System message to pass to the model
    system_message: str = """
    Assume you are a Business Analyst and you need to extract insights from the Database loaded into a variable 'df'. 
    The Database is a collection of data (Dimensions) to record the invoice Payment strategies of the customer. Managers and Analysts can use this data to do planning, understand the default customers, if the customer pays the invoices in time and reporting.
    
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
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
         {
            "column_name": "Customer", 
            "data_type": "object", 
            "sample_data": ['Amazon.com', 'Target Corp', 'Walmart Inc.'], 
            "column_description": "Customers is a business partner with whom business relationship exists."
        },
        {
            "column_name": "Strategy",
            "data_type": "object", 
            "sample_data": ['GEN001'], 
            "column_description": "Collection Strategy for the particular customer."
        },
        
        {
            "column_name": "Ontimepay %",
            "data_type": "object", 
            "sample_data": ['28.571429', '16.666667'], 
            "column_description": "Payments made on or before the due date.Any payment made on or before the due date is considered acceptable.If Column 'Arrears by Net Due Date' Values is '0' OR less than Zero i.e Payment is made on time. On-Time Payment Percentage is calculated as (Number of On-Time Payments / Total Number of Payments) * 100%.."
        },
        {
            "column_name": "Delay %", 
            "data_type": "object", 
            "sample_data": ['71.428571', '83.333333'], 
            "column_description": "Delays in payments beyond the agreed payment terms.Delays of no more than X days (e.g., 15 days) are considered acceptable. If Column 'Arrears by Net Due Date' Values is GE '0'  i.e Payment is delayed. Delay Percentage is calculated as (Number of Delayed Payments / Total Number of Payments) * 100%."
        },
        {
            "column_name": "OverdueInvoice%", 
            "data_type": "object", 
            "sample_data": ['51.851852', '88.888889	'], 
            "column_description": "Invoices that remain unpaid past their due dates.Threshold: No more than (e.g., 5%) of invoices should be overdue at any given time. A higher percentage may indicate potential issues."
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
            "column_name": "overdue_invoice_sum", 
            "data_type": "float64", 
            "sample_data": [1520.12, 691.12, 3040.25], 
            "column_description": "The total of overdue invoice sum "
        },
        {
            "column_name": "overdue_invoice_sum_30day", 
            "data_type": "float64", 
            "sample_data": [799.46, 841.54, 1351.03], 
            "column_description": "The total of overdue invoice sum, if the net overdue days are less than or equal to 3days "
        },
        {
            "column_name": "Risk Class", 
            "data_type": "float64", 
            "sample_data": ['A', 'B', 'C', 'D', 'E'], 
            "column_description": "Risk class assigned to the customer based on score.The different classes which indicates low, high risk levels"
        },
         {
            "column_name": "valu_1", 
            "data_type": "float64", 
            "sample_data": [20, 0], 
            "column_description": "If the Risk class 'C', 'D', 'E' the value is 20 else 0"
        },
         {
            "column_name": "valu_2", 
            "data_type": "float64", 
            "sample_data": [35, 0], 
            "column_description": "If the overdue_invoice_sum value is between 50000 and 100000 and the Net Due Date is greater than todays date"
        },
         {
            "column_name": "valu_3", 
            "data_type": "float64", 
            "sample_data": [50, 0], 
            "column_description": "If the overdue_invoice_sum value is between 50000 and 100000 and the Net Due Date is greater than todays date - 30days. The valuation points in this "
        },
         {
            "column_name": "bp_ratio", 
            "data_type": "float64", 
            "sample_data": [0.26, 0.0, 0.3] 
            "column_description": "The Calculation made from the Balance sheet pdf. The formula is (Total current Assets - Inventories)/ Current Liabilities"
        },
        {
            "column_name": "Value Total", 
            "data_type": "float64", 
            "sample_data": ['20', '55'], 
            "column_description": "The total of Valuation points to see the priority of the customer."
        },    
        {
            "column_name": "Total current assets", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "It refers to the monetary value of a transaction in the primary currency used for reporting within a specific company code."
        },
        {
            "column_name": "Inventories", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "It refers to the monetary inventory with the customer."
        },
        {
            "column_name": "Total liabilities and stockholders’ equity", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "How much liabilities and Stockholders equity."
        },
        {
            "column_name": 'Status of Promise To Pay', 
            "data_type": "object", 
            "sample_data": ['Broken', 'Kept'], 
            "column_description": "Promises to pay are broken or kept."
        },
        {
            "column_name": 'Dispute Case ID', 
            "data_type": "object", 
            "sample_data": ['DC001', 'DC002', 'DC003', 'DC004', 'DC005', 'DC006', 'DC007'], 
            "column_description": "This column represents the ID of unique disputed cases."
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
            "column_name": "Business Partner", 
            "data_type": "object", 
            "sample_data": ['18', '21', '32'], 
            "column_description": "ID of each business partner with whom business relationship exists."
        },
         {
            "column_name": "Customer", 
            "data_type": "object", 
            "sample_data": ['Amazon.com', 'Target Corp', 'Walmart Inc.'], 
            "column_description": "Customers is a business partner with whom business relationship exists."
        },
    
        {
            "column_name": "Ontimepay %",
            "data_type": "object", 
            "sample_data": ['28.571429', '16.666667'], 
            "column_description": "Payments made on or before the due date.Any payment made on or before the due date is considered acceptable.If Column 'Arrears by Net Due Date' Values is '0' OR less than Zero i.e Payment is made on time. On-Time Payment Percentage is calculated as (Number of On-Time Payments / Total Number of Payments) * 100%.."
        },
        {
            "column_name": "Strategy",
            "data_type": "object", 
            "sample_data": ['GEN001'], 
            "column_description": "Collection Strategy for the particular customer."
        },
        {
            "column_name": "Delay %", 
            "data_type": "object", 
            "sample_data": ['71.428571', '83.333333'], 
            "column_description": "Delays in payments beyond the agreed payment terms.Delays of no more than X days (e.g., 15 days) are considered acceptable. If Column 'Arrears by Net Due Date' Values is GE '0'  i.e Payment is delayed. Delay Percentage is calculated as (Number of Delayed Payments / Total Number of Payments) * 100%."
        },
        {
            "column_name": "OverdueInvoice%", 
            "data_type": "object", 
            "sample_data": ['51.851852', '88.888889	'], 
            "column_description": "Invoices that remain unpaid past their due dates.Threshold: No more than (e.g., 5%) of invoices should be overdue at any given time. A higher percentage may indicate potential issues."
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
            "column_name": "overdue_invoice_sum", 
            "data_type": "float64", 
            "sample_data": [1520.12, 691.12, 3040.25], 
            "column_description": "The total of overdue invoice sum "
        },
        {
            "column_name": "overdue_invoice_sum_30day", 
            "data_type": "float64", 
            "sample_data": [799.46, 841.54, 1351.03], 
            "column_description": "The total of overdue invoice sum, if the net overdue days are less than or equal to 3days "
        },
        {
            "column_name": "Risk Class", 
            "data_type": "float64", 
            "sample_data": ['A', 'B', 'C', 'D', 'E'], 
            "column_description": "Risk class assigned to the customer based on score.The different classes which indicates low, high risk levels"
        },
         {
            "column_name": "valu_1", 
            "data_type": "float64", 
            "sample_data": [20, 0], 
            "column_description": "If the Risk class 'C', 'D', 'E' the value is 20 else 0"
        },
         {
            "column_name": "valu_2", 
            "data_type": "float64", 
            "sample_data": [35, 0], 
            "column_description": "If the overdue_invoice_sum value is between 50000 and 100000 and the Net Due Date is greater than todays date"
        },
         {
            "column_name": "valu_3", 
            "data_type": "float64", 
            "sample_data": [50, 0], 
            "column_description": "If the overdue_invoice_sum value is between 50000 and 100000 and the Net Due Date is greater than todays date - 30days. The valuation points in this "
        },
         {
            "column_name": "bp_ratio", 
            "data_type": "float64", 
            "sample_data": [0.26, 0.0, 0.3] 
            "column_description": "The Calculation made from the Balance sheet pdf. The formula is (Total current Assets - Inventories)/ Current Liabilities"
        },
        {
            "column_name": "Value Total", 
            "data_type": "float64", 
            "sample_data": ['20', '55'], 
            "column_description": "The total of Valuation points to see the priority of the customer."
        },    
        {
            "column_name": "Total current assets", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "It refers to the monetary value of a transaction in the primary currency used for reporting within a specific company code."
        },
        {
            "column_name": "Inventories", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "It refers to the monetary inventory with the customer."
        },
        {
            "column_name": "Total liabilities and stockholders’ equity", 
            "data_type": "float64", 
            "sample_data": [146791.0, 136221.0, 0, 0], 
            "column_description": "How much liabilities and Stockholders equity."
        },
        {
            "column_name": 'Status of Promise To Pay', 
            "data_type": "object", 
            "sample_data": ['Broken', 'Kept'], 
            "column_description": "Promises to pay are broken or kept."
        },
        {
            "column_name": 'Dispute Case ID', 
            "data_type": "object", 
            "sample_data": ['DC001', 'DC002', 'DC003', 'DC004', 'DC005', 'DC006', 'DC007'], 
            "column_description": "This column represents the ID of unique disputed cases."
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