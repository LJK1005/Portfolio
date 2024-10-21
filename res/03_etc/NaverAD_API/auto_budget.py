import time
import requests
import pandas as pd
import datetime

import hashlib
import hmac
import base64


class Signature:

    @staticmethod
    def generate(timestamp, method, uri, secret_key):
        message = "{}.{}.{}".format(timestamp, method, uri)
        hash = hmac.new(bytes(secret_key, "utf-8"), bytes(message, "utf-8"), hashlib.sha256)

        hash.hexdigest()
        return base64.b64encode(hash.digest())

BASE_URL = 'https://api.searchad.naver.com'
SECRET_KEY = '---'
API_KEY = '---'
CUSTOMER_ID = '---'

def get_header(method, uri, api_key, secret_key, customer_id):
    timestamp = str(round(time.time() * 1000))
    signature = Signature.generate(timestamp, method, uri, SECRET_KEY)
    return {'Content-Type': 'application/json; charset=UTF-8', 'X-Timestamp': timestamp, 'X-API-KEY': API_KEY, 'X-Customer': str(CUSTOMER_ID), 'X-Signature': signature}

uri = '/ncc/campaigns'
method = 'GET'
r = requests.get(BASE_URL + uri, headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))

budget_list = r.json()

df_ratio = pd.read_csv("./budget_ratio.csv")

df = pd.read_csv("./budget.csv")
df['date'] = pd.to_datetime(df['date'])
today = pd.to_datetime(datetime.date.today())
budget_amount = df.query('date == @today').iloc[0, 1]

for i in df_ratio.index:
    for j in budget_list:
        if j['nccCampaignId'] == df_ratio['id'][i]:
            target_campaign = j
            target_campaign['dailyBudget'] = int(round(budget_amount * df_ratio['ratio'][i], -4))
            # print(target_campaign['dailyBudget'])

            uri = '/ncc/campaigns/' + target_campaign['nccCampaignId']
            method = 'PUT'
            r = requests.put(BASE_URL + uri, params = {'fields' : 'budget'}, json = target_campaign, headers=get_header(method, uri, API_KEY, SECRET_KEY, CUSTOMER_ID))

            # print("response status_code = {}".format(r.status_code))
            # print("response body = {}".format(r.json()))