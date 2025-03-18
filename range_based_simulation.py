# Libraries
import numpy as np
import pandas as pd
#import seaborn as sns
import scipy.stats as st
import plotly.express as px
from datetime import datetime, timedelta
from scipy.integrate import simpson
from sklearn.mixture import GaussianMixture
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Union
import re
from faker import Faker
# from faker import providers
import rstr
from tqdm import tqdm
path = r'C:\Users\sandipto.sanyal\Accenture\Synthify - General\synthify_application_code\synthify_backend'
import sys
sys.path.append(path)
# from python_modules import utils as u
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

# Define a function to create a templatized prompt
def create_prompt(template,text_generation, domain, examples, n_shots=0):
    """
    Generate a prompt for text generation.
    :param template: The base template string with placeholders for domain-specific prompts.
    :param domain: The specific domain for the test data, e.g., 'case notes' or 'ticket notes'.
    :param examples: A list of example strings for N-shot prompting.
    :param n_shots: Number of examples to include in the prompt (0 for zero-shot prompting).
    :return: A formatted prompt string.
    """
    prompt = template.format(text_generation = text_generation, domain=domain)
    if n_shots > 0:
        examples_text = "\n".join(examples[:n_shots])
        prompt = f"{examples_text}\n{prompt}"
    return prompt

def generate_text(prompt, max_length=100, temperature=0.7, num_return_sequences=10,num_rows=1000):
    """
    Generate multiple texts using a pre-trained model.
    :param prompt: The input prompt for text generation.
    :param max_length: The maximum length of the generated text.
    :param temperature: Sampling temperature for diversity.
    :param num_return_sequences: The number of different outputs to generate.
    :param num_beams: Number of beams to use for beam search.
    :return: A list of generated texts.
    """
    loop_iter = int(num_rows/num_return_sequences)
    reli = []
    for i in range(loop_iter):
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, temperature=temperature)
      # print(result)
        for res in result:
            reli.append(res['generated_text'].split('Output: ')[-1])
    if num_rows%num_return_sequences!=0:
        num_return_sequences = int(num_rows%num_return_sequences)
        print('inifnrs',num_return_sequences)
        result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, temperature=temperature)
      # print(result)
        for res in result:
            reli.append(res['generated_text'].split('Output: ')[-1])
    # print(res['generated_text'].split('Output: ')[-1])
    # print(reli)
    return reli

def fake_name(num_rows,language,region):
    if region and language:
        try:
            fake = Faker(f'{language}_{region}')
        except AttributeError:
            print('Error: The provided Language and/or region are not present in Faker module. Using default.')
            fake = Faker()
    else:
        fake = Faker()
    list_name = []
    for i in range(num_rows):
        list_name.append(fake.name())
    return list_name

def fake_address(num_rows, language, region):
    if region and language:
        try:
            fake = Faker(f'{language}_{region}')
        except AttributeError:
            print('Error: The provided Language and/or region are not present in Faker module. Using default.')
            fake = Faker()
    else:
        fake = Faker()
    list_address = []
    for i in range(num_rows):
        list_address.append(fake.address())
    return list_address

# Define the mathematical Distributions
def mixture_of_gaussian(x:np.array, means: list, std_devs: list):
    weight, data = 1, 0 
    for idx in range(len(means)):
        data += 1/(std_devs[idx]*np.sqrt(2*np.pi)) * np.exp(-(x - means[idx]) ** 2 / std_devs[idx]**2)
    return data

def uniform_distribution(min_val, max_val, num_rows):
    return np.random.uniform(min_val, max_val, num_rows)

def mixture_of_exponential(x: np.array, lmbdas: list, means: list):
    weight, data = 1, 0
    for idx in range(len(lmbdas)):
        data += np.exp(-lmbdas[idx] * (x - means[idx]))
    return data

def mixture_of_cauchy(x: np.array, scales: list, means: list):
    data = 0
    for idx in range(len(scales)):
        data += 1 / (np.pi * scales[idx] * (1 + ((x - means[idx]) / scales[idx]) ** 2))
    return data

def dates_distribution(min_date: str, max_date: str, mean_dates: List[str], std_days: List[dict]):
    '''
    Convert the dates into integer
    '''
    # checks which type of dateformat passed
    dts_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    dtsf_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3,6}$" 
    dt_pattern = r"^\d{4}-\d{2}-\d{2}$"
    t_pattern = r"\d{2}:\d{2}:\d{2}$"
    supported_patterns = [dts_pattern, dt_pattern, t_pattern]
    pattern = None
    if bool(re.match(dts_pattern, min_date)) and bool(re.match(dts_pattern, max_date)):
       pattern = '%Y-%m-%d %H:%M:%S'
    elif bool(re.match(dtsf_pattern, min_date)) and bool(re.match(dtsf_pattern, max_date)):
       pattern = '%Y-%m-%d %H:%M:%S.%f'
    elif bool(re.match(dt_pattern, min_date)) and bool(re.match(dt_pattern, max_date)):
       pattern = '%Y-%m-%d'
    elif bool(re.match(t_pattern, min_date)) and bool(re.match(t_pattern, max_date)):
       pattern = '%H:%M:%S'
    if not pattern:
       raise NotImplementedError('Only the following patterns are supported: {}'.format(supported_patterns))
    
    min_date = datetime.strptime(min_date, pattern)
    max_date = datetime.strptime(max_date, pattern)
    epoch_time = datetime(1970, 1, 1, 0, 0, 0, 0)
    if pattern == '%H:%M:%S':
       epoch_time = datetime(1900, 1, 1, 0, 0, 0)
    delta_min = int((min_date - epoch_time).total_seconds())
    delta_max = int((max_date - epoch_time).total_seconds())
    
    if mean_dates and std_days:
        delta_means, delta_stds = [], []
        for idx in range(len(mean_dates)):
            mean_date = datetime.strptime(mean_dates[idx], pattern)
            std_date = mean_date + relativedelta(**std_days[idx])
            delta_mean = int((mean_date - epoch_time).total_seconds())
            delta_std = int(((std_date - epoch_time).total_seconds()) - delta_mean)
            delta_means.append(delta_mean)
            delta_stds.append(delta_std)
        return delta_min, delta_max, delta_means, delta_stds, pattern
    return delta_min, delta_max, None, None, pattern


class gaussian(st.rv_continuous):
  def __init__(self, x:np.array, means: list, std_devs: list, **kwargs):
    super().__init__(**kwargs)
    self.means = means
    self.std_devs = std_devs
    self.const = simpson(mixture_of_gaussian(x, self.means, self.std_devs), x)

  def _pdf(self, x):
    return (1.0 / self.const) * mixture_of_gaussian(x, self.means, self.std_devs)

class exponential(st.rv_continuous):
  def __init__(self, x:np.array, lmbdas: list, means: list, **kwargs):
    super().__init__(**kwargs)
    self.lmbdas = lmbdas
    self.means = means
    self.const = simpson(mixture_of_exponential(x, self.lmbdas, self.means), x)

  def _pdf(self, x):
    return (1.0 / self.const) * mixture_of_exponential(x, self.lmbdas, self.means)

class cauchy(st.rv_continuous):
  def __init__(self, x:np.array, scales: list, means: list, **kwargs):
    super().__init__(**kwargs)
    self.scales = scales
    self.means = means
    self.const = simpson(mixture_of_cauchy(x, self.scales, self.means), x)

  def _pdf(self, x):
    return (1.0 / self.const) * mixture_of_cauchy(x, self.scales, self.means)
  
# convert the rvs from integer to datetime
def seconds_from_epoch_to_datetime(seconds:int, pattern:str):
    ep = datetime(1970,1,1,0,0,0,0)
    # milis = seconds % 24*3600*365*1000
    milis = 0
    final_date = ep + timedelta(seconds=int(seconds), milliseconds=milis)
    return final_date.strftime(pattern)

def mixture_of_gaussian_dates(min_date: str, max_date: str, means: List[str], stds: List[str], num_rows: int):
    # Define the means and standard deviations for the two Gaussian distributions
    # print(min_date, max_date, means, stds)
    list_distributions = []
    delta_min, delta_max, delta_means, delta_stds, pattern = dates_distribution(min_date, max_date, means, stds)
    # print(delta_min, delta_max, delta_means, delta_stds)
    x = np.linspace(delta_min, delta_max, num_rows)
    my_bb = gaussian(x=x, a=delta_min, b=delta_max, means=delta_means,
                     std_devs=delta_stds
                     )
    my_rvs =  my_bb.rvs(size=num_rows)
    # convert the rvs from integer to datetime
    # create a numpy vectorizer function to apply across all the elements in rv
    # datetime_from_seconds = np.vectorize(seconds_from_epoch_to_datetime)
    # # perform the transformation
    # my_rvs_datetime = datetime_from_seconds(my_rvs)

    return my_rvs, pattern

# Define the data generation function
def get_synthetic_data(num_rows, 
                       input_features:List[Dict[str, Union[str, int, List[int]]]]) -> pd.DataFrame:
    synthetic_data = {}
    # Generate synthetic numerical data
    for item in tqdm(input_features, 'Generating Samples:'):
        feature = item.get('col_name')
        distribution = item.get('distribution', 'uniform')
        mean = item.get('mean', None)
        mode = item.get('mode', None)
        std = item.get('std', None)
        min_val = item.get('min', None)
        max_val = item.get('max', None)
        dtype = item.get('dtype', None)
        # for id type columns
        pattern = item.get('pattern', 'auto_increment')

        if distribution == 'name':
            region = item.get('region','')
            language = item.get('language','')
            list_name = fake_name(num_rows=num_rows,language=language,region=region)
            synthetic_data[feature] = list_name

        elif distribution == 'address':
            region = item.get('region','')
            language = item.get('language','')
            list_address = fake_address(num_rows=num_rows,language=language,region=region)
            synthetic_data[feature] = list_address

        elif distribution == 'generated_text':

            template = item.get('template')
            text_generation = item.get('text_generation')
            domain_example = item.get('domain_example')
            n_shots = item.get('n_shots')
            domain = item.get('domain')
            examples = domain_example[domain]

            prompt = create_prompt(template, text_generation, domain, examples, n_shots)

            # Add the specific context for test data generation
            context = item.get('context')
            prompt = f"{prompt}\nContext: {context}\nOutput:"
            max_length = item.get('max_length')
            temperature = item.get('temperature')
            num_return_sequences = item.get('num_return_sequences')
            # Generate 10 different sentences based on the same prompt using beam search
            generated_texts = generate_text(prompt, max_length=max_length, temperature=temperature, num_return_sequences=num_return_sequences,num_rows=num_rows)
            synthetic_data[feature] = generated_texts
        elif distribution == 'uniform':
            if min_val and max_val and dtype:
                data = uniform_distribution(min_val, max_val, num_rows)
                if dtype == 'integer':
                    synthetic_data[feature] = data.astype(int)
                else:
                    synthetic_data[feature] = data

        # elif distribution == 'right_skewed':
        #     if min_val and max_val and dtype:
        #         mid = (min_val + max_val) / 2
        #         if mode is None:
        #             mode = (min_val + mid) / 2
        #         data = np.random.triangular(min_val, mode, max_val, size=num_rows)
        #         if dtype == 'integer':
        #             synthetic_data[feature] = data.astype(int)
        #         else:
        #             synthetic_data[feature] = data

        # elif distribution == 'left_skewed':
        #     if min_val and max_val and dtype:
        #         mid = (min_val + max_val) / 2
        #         if mode is None:
        #             mode = (mid + max_val) / 2
        #         data = np.random.triangular(min_val, mode, max_val, size=num_rows)
        #         if dtype == 'integer':
        #             synthetic_data[feature] = data.astype(int)
        #         else:
        #             synthetic_data[feature] = data

        elif distribution == 'mixture_gaussians':
            if min_val and max_val and dtype:
                x = np.linspace(min_val, max_val, num_rows)
                # print(min_val, max_val, mean, std)
                data_obj = gaussian(x=x, means=mean, std_devs=std, a=min_val, b=max_val)
                data_grvs = data_obj.rvs(size=num_rows)
                if dtype == 'integer':
                    synthetic_data[feature] = data_grvs.astype(int)
                else:
                    synthetic_data[feature] = data_grvs

        elif distribution == 'mixture_exponentials':
            lmbdas = item.get('lmbdas', None)
            if min_val and max_val and lmbdas and mean and dtype:
                x = np.linspace(min_val, max_val, num_rows)
                data_obj_e = exponential(x=x, lmbdas=lmbdas, means=mean, a=min_val, b=max_val)
                data_ervs = data_obj_e.rvs(size=num_rows)
                if dtype == 'integer':
                    synthetic_data[feature] = data_ervs.astype(int)
                else:
                    synthetic_data[feature] = data_ervs

        elif distribution == 'gaussian_date':
            if min_val and max_val and mean and std:
                date_distribution, pattern = mixture_of_gaussian_dates(min_val, max_val, mean, std, num_rows)
                datetime_from_seconds = np.vectorize(lambda x: seconds_from_epoch_to_datetime(x, pattern=pattern))
                # date_from_datetime = np.vectorize(lambda elem: elem.date())
                # time_from_datetime = np.vectorize(lambda elem: elem.time())
                gaussian_dates = datetime_from_seconds(date_distribution.reshape(-1).astype(int))
                synthetic_data[feature] = gaussian_dates.astype('str')

        elif distribution == 'uniform_date':
            if min_val and max_val:
                delta_min, delta_max, _, _, pattern = dates_distribution(min_val, max_val, None, None)
                epoch_seconds_range = uniform_distribution(delta_min, delta_max, num_rows).astype(int)
                vfunc = np.vectorize(lambda x:seconds_from_epoch_to_datetime(x, pattern=pattern))
                dates_u = vfunc(epoch_seconds_range)
                synthetic_data[feature] = dates_u

        elif distribution == 'manual_categorical':
            values = item.get('values', [])
            probabilities = item.get('probabilities', None)
            synthetic_data[feature] = np.random.choice(values, num_rows, p=probabilities)
        
        elif distribution == 'id':
            # get the pattern
            if pattern == 'auto_increment':
                if min_val:
                    values = np.arange(min_val, num_rows+min_val, 1, dtype='int')
                else:
                    values = np.arange(1, num_rows+1, 1)
            else: # pattern is a regex expression
                values = np.full(shape=num_rows, fill_value=pattern)
                convert_to_regex = np.vectorize(rstr.xeger)
                values = convert_to_regex(values)
            synthetic_data[feature] = values

        else:
            raise ValueError(f"Unsupported distribution for numerical feature '{feature}'")
 
    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df

def api_layer(num_rows, 
              input_features:List[Dict[str, Union[str, int, List[int]]]],
              user_id:str,
              dataset_name:str
              ):
    out:pd.DataFrame = get_synthetic_data(num_rows, input_features)
    # put the data in user tmp folder for download
    u.dump_var(user_id=user_id, var=out, name=dataset_name+"_sampled_df.df")
    u.dump_var(user_id=user_id, var=dataset_name, name='file.str')
    print(f'Dumped {dataset_name+".df"}')

    

if __name__ == '__main__':

    input_features = [
                # {"col_name": "UserID", "distribution":"id", "pattern":"auto_increment", "min":1000},
                # {"col_name": "Transaction_Id", "distribution": "id", "pattern":"^[A-Z0-9]{12,20}$"},
                # {"col_name": "Height", "dtype": "integer", "distribution": "uniform", "min": 140, "max": 200},
                # {"col_name": "Height2", "dtype": "integer", "distribution": "mixture_gaussians", "min": 15000, "max": 150000, "mean": [40000, 75000], "std": [5000, 3000]},
                # {"col_name": "Income", "dtype": "float", "distribution": "mixture_gaussians", "min": 15000, "max": 150000, "mean": [40000, 75000], "std": [5000, 3000]},
                # {"col_name": "Gender", "dtype": "categorical", "distribution": "manual_categorical", "values": ["Male", "Female"], "probabilities": [0.6, 0.4]},
                # {"col_name": "Education", "dtype": "categorical", "distribution": "manual_categorical", "values": ["High School", "Bachelor", "Master", "PhD"]},
                # {"col_name": "Emissions", "dtype": "float", "distribution": "mixture_exponentials", "min": 7, "max": 20, "lmbdas": [0.15, 0.25, 0.25], "mean": [5, 10, 25]},
                # {"col_name": "Date_Normal", "dtype": "date", "distribution": "gaussian_date", "min": "2018-01-01", "max": "2020-12-31", "mean": ["2019-06-01", "2020-01-01"], "std": [{"days":25}, {"days":30}]},
                # {"col_name": "DateTimestamp_Uniform", "dtype": "datetime", "distribution": "uniform_date", "min": "2023-02-04 00:00:00", "max": "2023-02-04 02:59:59",},
                # {"col_name": "DateTimestamp_Normal", "dtype": "datetime", "distribution": "gaussian_date", "min": "2023-02-04 00:00:00.356", "max": "2023-02-04 02:59:59.785", "mean": ["2023-02-04 01:00:00.000", "2023-02-04 01:30:30.000"], "std": [{"minutes":2, "hours":1}, {"seconds":58, "minutes":5}]},
                # {"col_name": "Time_Normal", "dtype": "time", "distribution": "gaussian_date", "min": "00:00:00", "max": "02:59:59", "mean": ["01:00:00"], "std": [{"minutes":15}]},
                # {"col_name": "Data_Text", "dtype": "str", "distribution": "generated_text",'template':"Generate a detailed {text_generation} for {domain} based on the following context:",'text_generation':"case notes",'domain_example':{'health':['Example 1: The patient reported severe abdominal pain and nausea after eating seafood.'],'legal':['Example 1: Whether a manufacturer owes a duty of care to a consumer who is not in a direct contractual relationship with them.', 'Example 2: The customer complained about delayed shipping and a damaged package.']},'n_shots':2,'context':"Related to Manufacturing industry.",'domain':"legal",'max_length':100, 'temperature':0.7, 'num_return_sequences':4 },
                {"col_name": "Name", "distribution":"name", "dtype":"str","language":'en', "region":"US"},
                {"col_name": "Address", "distribution":"address","dtype":"str","language":'en', "region":"US"}
                  ]

    out = get_synthetic_data(40, input_features)
    print(out)