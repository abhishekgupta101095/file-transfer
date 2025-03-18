import streamlit as st
from streamlit_chat import message
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
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

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
    system_message = """
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
        
    #Database Details:
    
    [
        {
            "column_name": "Version___Description", 
            "data_type": "object", 
            "sample_data": ['Actual', 'Plan'], 
            "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
        },
        {
            "column_name": "Date___FISCAL_CALPERIOD", 
            "data_type": "object", 
            "sample_data": ['202201', '202202', '202203'], 
            "column_description": "The date of the data that is being analyzed. This can be used to filter data or to analyze trends over time. For example, you might want to see how sales have changed over the past year."
        },
        {
            "column_name": "FLOW_TYPE___Description", 
            "data_type": "object", 
            "sample_data": ['Changing Amount', 'Closing Amount', 'Opening Amount', 'Control Parameter', 'Account Specific Growth Rate'], 
            "column_description": "The type of transaction that is being analyzed. This can be used to track different types of activities, such as Opening Balance, Closing Balance etc."
        },
        {
            "column_name": "SAP_ALL_COMPANY_CODE___Description", 
            "data_type": "object", 
            "sample_data": ['Company Code 1710'], 
            "column_description": "The company code can be used to track data for different business units or subsidiaries. For example, you might want to see how much money has been made by each of your company's subsidiaries such as Germany, India, US etc."
        },
        {
            "column_name": "SAP_FI_IFP_GLACCOUNT___Description", 
            "data_type": "object", 
            "sample_data": ['Bank 1 - Bank (Main) Account', 'Short Term Investments', 'Inventory - Raw Material', 'Inventory - Work In Progress', 'Inventory - Finished Goods', 'Machinery & Equipment', 'Computer Hardware and Equipment', 'Accumulated Depreciation - Machinery & Equipment', 'Accumulated Depreciation - Computer Hardware', 'Other Long Term Investments', 'Accumulated Amortization - Intangible Assets', 'Loans from Banks (no recon acct)', 'Pension Provision', 'Revenue Domestic - Product', 'Revenue Foreign - Product', 'Sales Rebates', 'Customer Sales Deduction - Domestic', 'COGS Direct Material', 'COGS Third Party', 'COGS Personnel Time', 'COGS Machine Time', 'COGS Production Overhead', 'Consumption - Raw Material', 'Adjustment Plant Activity Production Order', 'Travel Expenses - Miscellaneous', 'Payroll Expense - Salaries', 'Depreciation Expense - Machinery & Equipment', 'Depreciation Expense - Computer Hardware', 'Depreciation Expense - Intangible Assets', 'Profit for the period', 'Depreciation of property, plant and equipment', 'Amortization of intangible assets', 'Increase (Decrease) of provisions', 'Increase (Decrease) of inventories', 'Increase (Decrease) in other receivables (net)', 'Purchase (Sale) of tangible assets', 'Depreciation of tangible assets', 'Purchase (Sale) of intangible assets', 'Increase (Decrease) in long-term investments', 'Increase (Decrease) in notes receivable', 'Cash Flow Validation', 'Validation Balance', 'Receivables Domestic', 'Buildings', 'Payables Domestic', 'Accrued Net Payroll', 'Common Stock', 'Retained Earnings'], 
            "column_description": "The general ledger account that is being analyzed. This can be used to track financial transactions or to analyze costs and expenses. For example, you might want to see how much money has been spent on sales in a particular month."
        },
        {
            "column_name": "SAP_ALL_PROFITCENTER___Description", 
            "data_type": "object", 
            "sample_data": ['Consulting Unit A', 'Bike Parts', 'Bicycles', 'Shared Services', 'Trading Goods', 'Dummy Text', 'Unassigned'], 
            "column_description": "The profit center can be used to track data for different product lines or customer segments. For example, you might want to see how much profit has been made by each of your product lines."
        },
        {
            "column_name": "SAP_ALL_FUNCTIONALAREA___Description", 
            "data_type": "object", 
            "sample_data": ['Unassigned', 'Sales Revenue', 'Sales discounts and allow', 'Cost of Goods Sold', 'Consulting/Services', 'Sales and Distribution', 'Administration', 'Production'], 
            "column_description": "The functional area can be used to track data for different departments or business processes. For example, you might want to see how much money has been spent on marketing in a particular month."
        },
        {
            "column_name": "SAP_ALL_TRADINGPARTNER___Description", 
            "data_type": "object", 
            "sample_data": ['Unassigned', 'Company 1010'], 
            "column_description": "The trading partner can be used to track data for different Inter Company postings – e.g selling product from one company code and other company code buys it."
        },
        {
            "column_name": "AMOUNT", 
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

    system_message2 = """
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
            "column_name": "Version___Description", 
            "data_type": "object", 
            "sample_data": ['Actual', 'Plan'], 
            "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
        },
        {
            "column_name": "Date___FISCAL_CALPERIOD", 
            "data_type": "object", 
            "sample_data": ['202201', '202202', '202203'], 
            "column_description": "The date of the data that is being analyzed. This can be used to filter data or to analyze trends over time. For example, you might want to see how sales have changed over the past year."
        },
        {
            "column_name": "FLOW_TYPE___Description", 
            "data_type": "object", 
            "sample_data": ['Changing Amount', 'Closing Amount', 'Opening Amount', 'Control Parameter', 'Account Specific Growth Rate'], 
            "column_description": "The type of transaction that is being analyzed. This can be used to track different types of activities, such as Opening Balance, Closing Balance etc."
        },
        {
            "column_name": "SAP_ALL_COMPANY_CODE___Description", 
            "data_type": "object", 
            "sample_data": ['Company Code 1710'], 
            "column_description": "The company code can be used to track data for different business units or subsidiaries. For example, you might want to see how much money has been made by each of your company's subsidiaries such as Germany, India, US etc."
        },
        {
            "column_name": "SAP_FI_IFP_GLACCOUNT___Description", 
            "data_type": "object", 
            "sample_data": ['Bank 1 - Bank (Main) Account', 'Short Term Investments', 'Inventory - Raw Material', 'Inventory - Work In Progress', 'Inventory - Finished Goods', 'Machinery & Equipment', 'Computer Hardware and Equipment', 'Accumulated Depreciation - Machinery & Equipment', 'Accumulated Depreciation - Computer Hardware', 'Other Long Term Investments', 'Accumulated Amortization - Intangible Assets', 'Loans from Banks (no recon acct)', 'Pension Provision', 'Revenue Domestic - Product', 'Revenue Foreign - Product', 'Sales Rebates', 'Customer Sales Deduction - Domestic', 'COGS Direct Material', 'COGS Third Party', 'COGS Personnel Time', 'COGS Machine Time', 'COGS Production Overhead', 'Consumption - Raw Material', 'Adjustment Plant Activity Production Order', 'Travel Expenses - Miscellaneous', 'Payroll Expense - Salaries', 'Depreciation Expense - Machinery & Equipment', 'Depreciation Expense - Computer Hardware', 'Depreciation Expense - Intangible Assets', 'Profit for the period', 'Depreciation of property, plant and equipment', 'Amortization of intangible assets', 'Increase (Decrease) of provisions', 'Increase (Decrease) of inventories', 'Increase (Decrease) in other receivables (net)', 'Purchase (Sale) of tangible assets', 'Depreciation of tangible assets', 'Purchase (Sale) of intangible assets', 'Increase (Decrease) in long-term investments', 'Increase (Decrease) in notes receivable', 'Cash Flow Validation', 'Validation Balance', 'Receivables Domestic', 'Buildings', 'Payables Domestic', 'Accrued Net Payroll', 'Common Stock', 'Retained Earnings'], 
            "column_description": "The general ledger account that is being analyzed. This can be used to track financial transactions or to analyze costs and expenses. For example, you might want to see how much money has been spent on sales in a particular month."
        },
        {
            "column_name": "SAP_ALL_PROFITCENTER___Description", 
            "data_type": "object", 
            "sample_data": ['Consulting Unit A', 'Bike Parts', 'Bicycles', 'Shared Services', 'Trading Goods', 'Dummy Text', 'Unassigned'], 
            "column_description": "The profit center can be used to track data for different product lines or customer segments. For example, you might want to see how much profit has been made by each of your product lines."
        },
        {
            "column_name": "SAP_ALL_FUNCTIONALAREA___Description", 
            "data_type": "object", 
            "sample_data": ['Unassigned', 'Sales Revenue', 'Sales discounts and allow', 'Cost of Goods Sold', 'Consulting/Services', 'Sales and Distribution', 'Administration', 'Production'], 
            "column_description": "The functional area can be used to track data for different departments or business processes. For example, you might want to see how much money has been spent on marketing in a particular month."
        },
        {
            "column_name": "SAP_ALL_TRADINGPARTNER___Description", 
            "data_type": "object", 
            "sample_data": ['Unassigned', 'Company 1010'], 
            "column_description": "The trading partner can be used to track data for different Inter Company postings – e.g selling product from one company code and other company code buys it."
        },
        {
            "column_name": "AMOUNT", 
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
    
    # Output Format:
    Deduction:              Based on the conversations, we deduce that there is no Data for 'Plan' present in the DataFrame for the year 2023.
    Recommended Action:     Please change the year or add data for 'Plan' for the year 2023
    """




    user_query: Optional[str]
    code0 = """
import json
import pandas as pd
from astropy import units as u

# Open the json file
with open('raw_data_mod.json') as f:
    # Load the json object as a dictionary
    data = json.load(f)
    
df = pd.DataFrame(data["value"])
df = df.dropna()
df['DT_CONV']  = pd.to_datetime(df['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
df['QUARTER'] = df['DT_CONV'].dt.quarter

columns = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description', \
        'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', \
        'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',\
        'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT','QUARTER']

df = df[columns]"""

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

def answer_question(system_message, system_message2, user_query, code0):
    # As of now we are allowing 5 iterations

    # Encode the query with the model
    query_embedding = model.encode(user_query)

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

    # Set value of k for number of examples you want
    top_results = torch.topk(cosine_scores, k=1)

    # Update the system message as follows:
    i = 0
    top_1_solution = ''
    for indic in top_results[1]:
        top_1_solution = top_1_solution + solution[int(indic)]
        i = i + 1
    system_message_ex = system_message + top_1_solution

    payload = [{"role":"system","content":system_message},{"role":"user","content":user_query}]
    payload_ex = [{"role": "system", "content": system_message_ex}, {"role": "user", "content": user_query}]
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
                    output = generate_response(payload_ex)
                #print(output)
                try:
                    #code = "#"+output.split("```")[1]
                    matches = re.findall(r"```([\s\S]*?)```", output)
                    code = "#"+" \n\n".join(matches)
                except:
                    code = "#"

                exec(code0+code)
                try:
                    final_answer = pd.read_csv('output.csv')
                    df = pd.DataFrame({})
                    df.to_csv('output.csv', index=False)
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

def app():
    # Open the json file
    with open('raw_data_mod.json') as f:
        # Load the json object as a dictionary
        data = json.load(f)

    # Convert into a Dataframe    
    df = pd.DataFrame(data["value"])
    df = df.dropna() 
    columns = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description',
            'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description',
            'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',
            'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT']

    df = df[columns]

    #Separating year and month column.
    df['YEAR']=df['Date___FISCAL_CALPERIOD'].str.slice(0,4)
    df['MONTH']=df['Date___FISCAL_CALPERIOD'].str.slice(4,6)
    df['DT_CONV']  = pd.to_datetime(df['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
    df['QUARTER'] = df['DT_CONV'].dt.quarter
    filter_df = df



    # Setting title and page layout
    #st.set_page_config(layout ="wide")
    st.title('SAP FINANCE INSIGHTS')

    #To select the database
    st.sidebar.selectbox('Select the database:',['SAC1','SAC2'])

    #A dictionary to map laymen terms to actual column name
    dic = {'Version': 'Version___Description', 'Date': 'Date___FISCAL_CALPERIOD',
           'Flow Type': 'FLOW_TYPE___Description',
            'GL Account': 'SAP_FI_IFP_GLACCOUNT___Description',
           'Functional Area': "SAP_ALL_FUNCTIONALAREA___Description",
           'Trading Partner': "SAP_ALL_TRADINGPARTNER___Description", 'Amount': "AMOUNT"}

    #Checkbox set to include entire dataframe as default
    all = st.sidebar.checkbox("Selected all columns (Uncheck the box for specific columns).", value=True)

    #If user want to study on specific columns
    if all==False:
        selected_columns = st.sidebar.multiselect("Select one or more Columns:", list(dic.keys()))
        li=['YEAR','QUARTER','MONTH','SAP_ALL_PROFITCENTER___Description','SAP_ALL_COMPANY_CODE___Description']
        for i in selected_columns:
            li.append(dic[i])
        filter_df = filter_df[li]
    else:
        li = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description',
            'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description',
            'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',
            'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT','YEAR','QUARTER','MONTH']
        filter_df = filter_df[li]

    global selected_quarters
    global selected_years
    global selected_months
    global year

    selected_years= list(filter_df['YEAR'].drop_duplicates())
    selected_quarters=list(filter_df['QUARTER'].drop_duplicates())
    selected_months=list(filter_df['MONTH'].drop_duplicates())

    year='2022'

    selected_pc=st.sidebar.multiselect('Select the products:', filter_df['SAP_ALL_PROFITCENTER___Description'].drop_duplicates(), filter_df['SAP_ALL_PROFITCENTER___Description'].drop_duplicates())
    filter_df = filter_df.loc[filter_df['SAP_ALL_PROFITCENTER___Description'].isin(selected_pc)]
    selected_cc=st.sidebar.multiselect('Select all the company codes:',filter_df['SAP_ALL_COMPANY_CODE___Description'].drop_duplicates(),filter_df['SAP_ALL_COMPANY_CODE___Description'].drop_duplicates())
    filter_df = filter_df.loc[filter_df['SAP_ALL_COMPANY_CODE___Description'].isin(selected_cc)]

    #Select the type of analysis
    analysis=st.sidebar.selectbox('Select the type of analysis:',['<Select>','Comparative','Yearly'])
    if analysis == '<Select>':

        # Select years to analyse (Compulsory)
        selected_years = st.sidebar.multiselect("Select one or more year:", filter_df['YEAR'].drop_duplicates(),filter_df['YEAR'].drop_duplicates())
        filter_df = filter_df.loc[filter_df["YEAR"].isin(selected_years)]

        # By default all the quarters are selected
        allQuarter = st.sidebar.checkbox("Selected all Quarters (Uncheck for Quarter based analysis).", value=True)

        # If user want quarterly study
        if allQuarter == False:
            selected_quarters = st.sidebar.multiselect("Select one or more Quarter:", filter_df['QUARTER'].drop_duplicates())
            filter_df = filter_df[(filter_df["YEAR"].isin(selected_years)) & (filter_df["QUARTER"].isin(selected_quarters))]

        allMonth = st.sidebar.checkbox("Selected all Months (Uncheck to select any month).", value=True)

        # If user want monthly study
        if allMonth == False:
            selected_months = st.sidebar.multiselect("Select one or more Months:", filter_df['MONTH'].drop_duplicates())
            filter_df = filter_df[(filter_df["YEAR"].isin(selected_years)) & (filter_df["QUARTER"].isin(selected_quarters)) & (filter_df["MONTH"].isin(selected_months))]
        #st.write('Please select an analysis type')
    elif analysis == 'Comparative':
        # Do the comparative analysis
        st.write('Please select atleast one year for Comparative analysis.')

        #Select years to analyse (Compulsory)

        selected_years=st.sidebar.multiselect("Select one or more year:",filter_df['YEAR'].drop_duplicates())
        filter_df=filter_df.loc[filter_df["YEAR"].isin(selected_years)]

        #By default all the quarters are selected
        allQuarter = st.sidebar.checkbox("Selected all Quarters (Uncheck for Quarter based analysis).", value=True)

        #If user want quarterly study
        if allQuarter==False:

            selected_quarters=st.sidebar.multiselect("Select one or more Quarter:",filter_df['QUARTER'].drop_duplicates())
            filter_df = filter_df[(filter_df["YEAR"].isin(selected_years)) & (filter_df["QUARTER"].isin(selected_quarters))]

        allMonth = st.sidebar.checkbox("Selected all Months (Uncheck to select any month).", value=True)

        # If user want monthly study
        if allMonth == False:
            selected_months = st.sidebar.multiselect("Select one or more Months:", filter_df['MONTH'].drop_duplicates())
            filter_df = filter_df[(filter_df["YEAR"].isin(selected_years)) & (filter_df["QUARTER"].isin(selected_quarters)) & (filter_df["MONTH"].isin(selected_months))]


    elif analysis == 'Yearly':
        # Do the yearly analysis

        year=st.sidebar.selectbox('Select years', (filter_df['YEAR']).drop_duplicates())
        filter_df=filter_df[filter_df["YEAR"]==year]

    #To select version
    descr=st.sidebar.selectbox('Select the version description',['<Select>','Actual','Plan'])
    if descr == '<Select>':
        # Do nothing or display a message
        print('')
        #st.write('Please select a version description')
    elif descr == 'Actual':
        # Filter by actual version
        filter_df=filter_df[filter_df['Version___Description']=='Actual']
    elif descr == 'Plan':
        # Filter by plan version
        filter_df=filter_df[filter_df['Version___Description']=='Plan']

    # Checking the data and Querying the SAC Database
    st.info('chat with SAC Database')
    input_text = st.chat_input("Enter your query")
    
    if input_text:
        message(input_text, is_user=True)
        user_query = f"""
                Answer the following question: {input_text}
                """
        if selected_pc!= list(filter_df['SAP_ALL_PROFITCENTER___Description'].drop_duplicates()):
            user_query = user_query + f"""
                            For the 'SAP_ALL_PROFITCENTER___Description': {selected_pc}
                        """
        if selected_cc!= list(filter_df['SAP_ALL_COMPANY_CODE___Description'].drop_duplicates()):
            user_query = user_query + f"""
                            For the 'SAP_ALL_COMPANY_CODE___Description': {selected_cc}
                        """
        if all==False:
            user_query = user_query + f"""
                            For the Column(s): {li}
                        """
        if analysis!='<Select>':
            if analysis == 'Comparative':
                user_query = user_query+f"""
                    For the Year(s): {selected_years}
                    For the Quarter(s): {selected_quarters}
                    For the Month(s): {selected_months}
                """
            elif analysis == 'Yearly':
                user_query = user_query+f"""
                    For the Year(s): {year}
                """

        if descr!='<Select>':
            user_query = user_query + f"""
                    For the 'Version___Description': {descr}
                """
        output = answer_question(system_message, system_message2, user_query, code0)
        st.sidebar.title("Solution Panel")
        try:
            st.sidebar.write(output['solution'])
        except:
            st.sidebar.write('None')
        
        st.write(output['final_answer'])