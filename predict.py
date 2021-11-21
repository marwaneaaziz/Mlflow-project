import json 
import requests
import pandas as pd
from requests.api import head 

data = pd.read_csv('test_data.csv')
url = 'https//0.0.0.0.1234/invocations'
headers = {"ContentType":'text/csv'}
response = requests.post(url,data,headers=headers)
print(response.text)