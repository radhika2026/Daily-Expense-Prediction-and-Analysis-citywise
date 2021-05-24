import pandas as pd                   # reading csv files and dataframes
import numpy as np                    # matrix manipulation
from datetime import date
from geopy import geocoders          # getting country names from city names
import requests                       # getting exchange rates for currencies
import matplotlib.pyplot as plt       # plotting processed data
df = pd.read_csv('./expenses.csv')
print(df.iloc[90:130, :11])

def city_to_country(city):
    gn = geocoders.GeoNames("", "<---myUsername--->")
    return (gn.geocode(city)[0].split(", ")[2].lower())

def get_exchange_rate(base_currency, target_currency, date):
    if base_currency == target_currency:
        return 1
    date_formatted = "-".join(date[:-1].split('.')[::-1])
    api_uri = "https://free.currencyconverterapi.com/api/v6/convert?q={}&compact=ultra&date={}"\
        .format(base_currency + "_" + target_currency, date_formatted)
    api_response = requests.get(api_uri)
    if api_response.status_code == 200:
        return float(api_response.json()[base_currency+"_"+target_currency][date_formatted])

country_to_currency = {
        'croatia': 'HRK',
        'poland': 'PLN',
        'italy': 'EUR',
        'germany': 'EUR',
        'sweden': 'SEK',
        'denmark': 'DKK',
        'czechia': 'CZK',
        }

def transform_row(r):
    if len(r.date) == 6:
        r.date += '2018.'
    d = r.date[:-1].split('.')
    r.date = date(*map(int, d[::-1]))
    r.country = city_to_country(r.city)
    r.currency = country_to_currency[r.country]
    if np.isnan(r.hrk):
        r.hrk = r.lcy * get_exchange_rate(r.currency, 'HRK', r.date)
    r.eur = r.hrk * get_exchange_rate('HRK', 'EUR', r.date)
    return r

df = df.apply(transform_row, axis=1) # applying the function to each row
print(df.iloc[90:130, :11])

category_sum = []
for category, rows in df.groupby(['category'])['eur']:
    category_sum.append((sum(rows.values), category))
sums, labels = zip(*sorted(category_sum, reverse=True)[:11])
explode = [0.1] * len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('percentage of money spend on each category')
plt.show()

preferred_transport = []
for desc, rows in df.groupby(['description']):
    if all(i in ['travel', 'transport'] for i in rows['category']):
        preferred_transport.append((sum(rows['eur'].values), desc))

sums, labels = zip(*sorted(preferred_transport, reverse=True))
explode = [0.1]*len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('preferred public transport')
plt.show()
