import os
import json
import openai

import datetime as dt
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
from pandasai.llm.azure_openai import AzureOpenAI


## openai API keys
# load_dotenv('./.env')

# Openai Credentials
openai.api_type = "azure"
openai.api_base = "https://iats-earnings-openai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = '89f69c707099424bb559a95ccbad1a53'
#openai.verify_ssl_certs = False

llm = AzureOpenAI(api_token = openai.api_key,
                  api_base = openai.api_base,
                  api_version = openai.api_version,
                  api_type = openai.api_type,
                  deployment_name = "gpt-35-turbo")


# Function for pandas ai to query Tabular data
def chat_with_csv(df, prompt):
    #llm =OpenAI()  
    llm = AzureOpenAI(api_token = openai.api_key,
                  api_base = openai.api_base,
                  api_version = openai.api_version,
                  api_type = openai.api_type,
                  deployment_name = 'gpt-35-turbo')
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df, prompt=prompt)
    print(result)
    return result

# Setting title and page layout
st.set_page_config(layout ="wide")
st.title('SAP FINANCE INSIGHTS')

# Open the json file
with open(r"C:\Users\abhishek.cw.gupta\Downloads\raw_data\raw_data.json") as f:
    # Load the json object as a dictionary
    data = json.load(f)

# Convert into a Dataframe    
df = pd.DataFrame(data["value"])
data = df.dropna() 
columns = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description', \
         'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', \
         'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',\
         'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT']

data = data[columns]

#Separating year and month column.
data['YEAR']=data['Date___FISCAL_CALPERIOD'].str.slice(0,4)
data['MONTH']=data['Date___FISCAL_CALPERIOD'].str.slice(4,6)
data['DT_CONV']  = pd.to_datetime(df['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
data['QUARTER'] = data['DT_CONV'].dt.quarter

#To select the database
dataB=st.sidebar.selectbox('Select the database:',['SAC1','SAC2'])

#Select the type of analysis
analysis=st.sidebar.selectbox('Select the type of analysis:',['Select','Comparative','Yearly'])


#For comparative analysis
#year_list = data['YEAR'].drop_duplicates()
#all = st.checkbox("Select all", value=True)
#if all: 
#    selected_options = st.sidebar.multiselect("Select one or more years:", year_list, year_list)
    #data = data

if analysis=='Select':
    data=data   
    
elif analysis=='Comparative':
     selected_years=st.sidebar.multiselect("Select one or more year:",data['YEAR'].drop_duplicates())
     data=data.loc[data["YEAR"].isin(selected_years)]
     selected_quarters=st.sidebar.multiselect("Select one or more Quarter:",data['QUARTER'].drop_duplicates())
     data=data.loc[data["QUARTER"].isin(selected_quarters)]


#For yearly analysis
elif analysis=='Yearly':
    year=st.sidebar.selectbox('Select years', (data['YEAR']).drop_duplicates())
    st.sidebar.write('The selected year is ', year)
    data=data[data["YEAR"]==year]



#To select version
descr=st.sidebar.selectbox('Select the version description',['<Select>','Actual','Plan'])

#If filtered based on the Actual or plan
if descr=='<Select>':
    data=data
else:
    data=data[data['Version___Description']==descr]

st.write(data)    
# Checking the data and Querying the SAC Database
if data is not None:
    # col1, col2 = st.columns([1,1])   
    st.info('chat with SAC Database')      
    input_text = st.text_area("Enter your query")
    if input_text is not None:
        if st.button("Q&A with SAC"): 
            st.info("Your query :: " + input_text )
            result=chat_with_csv(data, input_text)
            st.success(result) 
else:
    st.write("No Data Found")                                


