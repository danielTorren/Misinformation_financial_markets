import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

#Importing the df

df = pd.read_csv("results\empirical_data_from2012-01-01to2022-12-01\Data\dividends.csv")
df = df.set_index('Date')


# df.loc[:,'AAPL'][df.loc[:,'AAPL'] > 0.0].plot()
# plt.show()



# #Plotting dividends
# for symbol in df.columns[1:3]:
#     plt.plot(df.index[df.loc[:,symbol] > 0.0], df.loc[:,symbol][df.loc[:,symbol] > 0.0], alpha = 1.0, linewidth=0.5)
    
    

    
# plt.show()