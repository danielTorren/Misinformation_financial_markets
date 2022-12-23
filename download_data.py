import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
from utility import (
    createFolder, 
    load_object, 
)

#Get the list of all S&P 500 stocks from wikipedia
resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]


#Define the starting and endate of our df (for now 10 years)
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2022, 12, 1)

# Getting data
data = yf.download(tickers, start=start, end=end, actions = True)
dividends = pd.DataFrame(pd.DataFrame(data.loc[:,'Dividends']))
prices = pd.DataFrame(pd.DataFrame(data.loc[:,'Adj Close']))
volume = pd.DataFrame(pd.DataFrame(data.loc[:,'Volume']))

#Saving
fileName = "results/empirical_data_from"+str(start)[0:10]+"to"+str(end)[0:10]
createFolder(fileName)
dividends.to_csv(fileName + "/Data" + "/dividends.csv")
prices.to_csv(fileName + "/Data" + "/prices.csv")
volume.to_csv(fileName + "/Data" + "/volume.csv")


