import streamlit as st
from streamlit_chat import message
#Note: The openai-python library support for Azure OpenAI is in preview.
import openai

openai.api_type = "azure"
openai.api_base = "https://iats-earnings-openai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "89f69c707099424bb559a95ccbad1a53"

system_message = """
Assume you are a Business Analyst and you need to extract insights from the Database loaded into a variable 'df'. 
The Database is a collection of data (Dimensions) to record the Planning and Actual numbers of various Profit centers. Project Managers and Analysts can use this data to do planning, budgeting, forecasting, and reporting.

Your Task is to do the following:
    * Based on the 'Database Details' and 'user question', provide the necessary code to obtain the results as shown in the 'Example'
    * Assume you have the dataframe already loaded into a variable 'df'.
    * The User will constantly be providing you with the output to your generated code. Use that as FEEDBACK to improve.
    * In case the User provides the expected response. Generate NO FURTHER CODE
    * Write your final response to 'output.txt'
    
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
        "sample_data": "Actual", 
        "column_description": "This is used for tracking changes over time, such as if you want to compare data from different versions of a model such as Actual V/s Plan comparison."
    },
    {
        "column_name": "Date___FISCAL_CALPERIOD", 
        "data_type": "object", 
        "sample_data": "202201", 
        "column_description": "The date of the data that is being analyzed. This can be used to filter data or to analyze trends over time. For example, you might want to see how sales have changed over the past year."
    },
    {
        "column_name": "FLOW_TYPE___Description", 
        "data_type": "object", 
        "sample_data": "Changing Amount", 
        "column_description": "The type of transaction that is being analyzed. This can be used to track different types of activities, such as Opening Balance, Closing Balance etc."
    },
    {
        "column_name": "SAP_ALL_COMPANY_CODE___Description", 
        "data_type": "object", 
        "sample_data": "Company Code 1710", 
        "column_description": "The company code can be used to track data for different business units or subsidiaries. For example, you might want to see how much money has been made by each of your company's subsidiaries such as Germany, India, US etc."
    },
    {
        "column_name": "SAP_FI_IFP_GLACCOUNT___Description", 
        "data_type": "object", 
        "sample_data": "Bank 1 - Bank (Main) Account", 
        "column_description": "The general ledger account that is being analyzed. This can be used to track financial transactions or to analyze costs and expenses. For example, you might want to see how much money has been spent on sales in a particular month."
    },
    {
        "column_name": "SAP_ALL_PROFITCENTER___Description", 
        "data_type": "object", 
        "sample_data": "Consulting Unit A", 
        "column_description": "The profit center can be used to track data for different product lines or customer segments. For example, you might want to see how much profit has been made by each of your product lines."
    },
    {
        "column_name": "SAP_ALL_FUNCTIONALAREA___Description", 
        "data_type": "object", 
        "sample_data": "Unassigned", 
        "column_description": "The functional area can be used to track data for different departments or business processes. For example, you might want to see how much money has been spent on marketing in a particular month."
    },
    {
        "column_name": "SAP_ALL_TRADINGPARTNER___Description", 
        "data_type": "object", 
        "sample_data": "Unassigned", 
        "column_description": "The trading partner can be used to track data for different Inter Company postings – e.g selling product from one company code and other company code buys it."
    },
    {
        "column_name": "AMOUNT", 
        "data_type": "float64", 
        "sample_data": -11000.0, 
        "column_description": "represents the transactional data Value"
    }
]

#Example:
    Question: What were the main drivers of cost increases in 2022

Output:
    To identify the main drivers of cost increases in 2022, we need to analyze the data further. However, based on the given column details, we can filter and group the data to get a preliminary understanding of the cost increases. Here's the code to do that:

    ```python
    # Filter the necessary columns
    df = df[['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description', 'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', 'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description', 'AMOUNT']]

    # Filter the data for the year 2022 only
    df_filtered = df[(df['Date___FISCAL_CALPERIOD'] >= '202201') & (df['Date___FISCAL_CALPERIOD'] <= '202212')]

    # Group the data by the necessary columns and get the total amount for each group
    grouped_data = df_filtered.groupby(['SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', 'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description']).sum().reset_index()

    # Sort the data by the amount in descending order
    sorted_data = grouped_data.sort_values(by='AMOUNT', ascending=False)

    # Print the top 10 rows to see the main drivers of cost increases
    print(sorted_data.head(10))
    with open('output.txt', 'w') as f:
        f.write(str(sorted_data.head(10)))
    ```

    This code will give us a grouped and sorted view of the data, which can help us identify the main drivers of cost increases in 2022. However, further analysis may be required to get a more accurate understanding of the cost increases.
"""

code0 = """
import json
import pandas as pd

# Open the json file
with open('raw_data.json') as f:
    # Load the json object as a dictionary
    data = json.load(f)
    
df = pd.DataFrame(data["value"])
df = df.dropna()

columns = ['Version___Description', 'Date___FISCAL_CALPERIOD', 'FLOW_TYPE___Description', \
        'SAP_ALL_COMPANY_CODE___Description', 'SAP_FI_IFP_GLACCOUNT___Description', \
        'SAP_ALL_PROFITCENTER___Description', 'SAP_ALL_FUNCTIONALAREA___Description',\
        'SAP_ALL_TRADINGPARTNER___Description', 'AMOUNT']

df = df[columns]


"""

def generate_response(payload):
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages = payload,
        temperature=0.3,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def answer_question(system_message, user_query, code0):
    # As of now we are allowing 18 iterations 9+3+3+3
    payload = [{"role":"system","content":system_message},{"role":"user","content":user_query}]
    reboot_message = """
                    SYSTEM MESSAGE: Sorry I am still learning and might go off course sometimes. 
                    Please allow me to retrace my steps to check what seems to be the issue.
                    """
    j=0
    while 1:
        counter = len(payload)/2
        if j==3:
            break
        try:
            if counter>9 and j<3:
                print(reboot_message)
                payload = [payload[0], payload[1], payload[2], payload[3], payload[-8], payload[-7], \
                        payload[-6], payload[-5], payload[-4], payload[-3], payload[-2], payload[-1]]
                counter = len(payload)/2
                j=j+1
            
    
            print("Iteration "+str(counter))
            output = generate_response(payload)
            #print(output)
            try:
                code = "#"+output.split("```")[1]
            except:
                code = "#"

            print(exec(code0+code))
            return {'solution': output}
        except Exception as e:
            error_msg = "ERROR: "+str(repr(e))
            print(error_msg)
            payload.append({"role": "assistant", "content": output})
            payload.append({"role": "user", "content": error_msg})

def app():
    # Define a list of databases
    databases = ['SAC1','SAC2']

    # Create a sidebar selectbox to choose a database
    selected_database = st.sidebar.selectbox("Select a database", databases)

    # Display the selected database
    st.title(f"Working with {selected_database}")

    # Create a chat input widget to get user questions
    user_query = st.chat_input("Ask a question")

    # If the user enters a question, send it to the answer_question function
    if user_query:
        # Display the user question as a chat message
        message(user_query, is_user=True)

        # Call the answer_question function and capture its output
        output = answer_question(system_message, user_query, code0)

        st.sidebar.title("Solution Panel")
        st.sidebar.write(output['solution'])

        with open('output.txt') as f:
            final_answer = f.read()

        # Display the output as a chat message
        message(final_answer)
