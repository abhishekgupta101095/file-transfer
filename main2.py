import json
import streamlit as st
from streamlit_chat import message
import pandas as pd
from pandasai.llm.azure_openai import AzureOpenAI
from pandasai import PandasAI
import openai

openai.api_type = "azure"
openai.api_base = "https://openai-sapaicx-canada.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = '57e7b0f37d0f4dc1b657496a726156c0'

llm = AzureOpenAI(api_token = openai.api_key,
                 api_base = openai.api_base,
               api_version = openai.api_version,
              api_type = openai.api_type,
               deployment_name = "gpt-35-turbo")

pandas_ai = PandasAI(llm)

with open('raw_data.json') as f:
    # Load the json object as a dictionary
    data = json.load(f)

df = pd.DataFrame(data['value'])

#Filtering the dataframe based on required columns.
df1 = df[['Version___Description','Date___FISCAL_CALPERIOD','FLOW_TYPE___Description','SAP_ALL_COMPANY_CODE___Description','SAP_FI_IFP_GLACCOUNT___Description',"SAP_ALL_PROFITCENTER___Description","SAP_ALL_FUNCTIONALAREA___Description","SAP_ALL_TRADINGPARTNER___Description","AMOUNT"]]

#Separating year and month column.
df1['YEAR']=df1['Date___FISCAL_CALPERIOD'].str.slice(0,4)
df1['MONTH']=df1['Date___FISCAL_CALPERIOD'].str.slice(4,6)

#To select the database
dataB=st.sidebar.selectbox('Select the database:',['SAC1','SAC2'])

#Select the type of analysis
analysis=st.sidebar.selectbox('Select the type of analysis:',['Comparative','Yearly'])

#For comparative analysis
if analysis=='Comparative':
    selected_years=st.sidebar.multiselect("Select one or more year:",df1['YEAR'].drop_duplicates())
    df2=df1.loc[df1["YEAR"].isin(selected_years)]

#For yearly analysis
else:
    year=st.sidebar.selectbox('Select years', (df1['YEAR']).drop_duplicates())
    st.sidebar.write('The selected year is ', year)
    df2=df1[df1["YEAR"]==year]

#To select version
descr=st.sidebar.selectbox('Select the version description',['<Select>','Actual','Plan'])

#If filtered based on the data
if descr!='<Select>':
    df3=df2[df2['Version___Description'] == descr]
    st.sidebar.write('The selected Description is ', descr)

#By default
else:
    df3=df2

user_query = st.chat_input("Ask a question")
if user_query:
    # Display the user question as a chat message
    message(user_query, is_user=True)

    # Call the answer_question function and capture its output
    output = pandas_ai(df3, prompt=user_query)
    message(output)