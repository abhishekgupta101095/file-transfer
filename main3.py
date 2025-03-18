import json
import pandas as pd

with open(r"C:\Users\abhishek.cw.gupta\Downloads\raw_data\raw_data.json") as f:
# Load the json object as a dictionary
    data = json.load(f)
df = pd.DataFrame(data['value'])

#Filtering the dataframe based on required columns.
df1 = df[['Version___Description','Date___FISCAL_CALPERIOD',\
          'FLOW_TYPE___Description','SAP_ALL_COMPANY_CODE___Description',
          'SAP_FI_IFP_GLACCOUNT___Description',"SAP_ALL_PROFITCENTER___Description",\
          "SAP_ALL_FUNCTIONALAREA___Description","SAP_ALL_TRADINGPARTNER___Description","AMOUNT"]]

#Separating year and month column.
df1['YEAR']=df1['Date___FISCAL_CALPERIOD'].str.slice(0,4)
df1['MONTH']=df1['Date___FISCAL_CALPERIOD'].str.slice(4,6)
df1['DT_CONV']  = pd.to_datetime(df1['Date___FISCAL_CALPERIOD'], format='%Y%m', errors='coerce').dropna()
df1['QUARTER'] = df1['DT_CONV'].dt.quarter

df1['Text']='In the fiscal year '+df1['YEAR']+' Quarter'+df1['QUARTER'].astype(str)+\
            ' ,for '+df1['SAP_ALL_COMPANY_CODE___Description']+' and profit centre '+df1["SAP_ALL_PROFITCENTER___Description"]+\
            ', the '+df1['Version___Description']+' amount is '+df1['AMOUNT'].astype(str)+', flow type for this is '+df1['FLOW_TYPE___Description']+\
            '. The description of GL Account used for transaction is '+df1['SAP_FI_IFP_GLACCOUNT___Description']+', the functional area is '\
            +df1["SAP_ALL_FUNCTIONALAREA___Description"]+', the trading partner is '+df1["SAP_ALL_TRADINGPARTNER___Description"]+'.'

print(df1['Text'][0])
df1.to_csv(r"C:\Users\abhishek.cw.gupta\Downloads\raw_data\Raw_Data.csv")