def GetDataYahoo(asset_list,start_dt,end_dt):
# DOWNLOAD YAHOO DATA 
# Version 0.0.1
# INPUTS
#   asset_list --> a series (dataframe) of one colums of tickers in Yahoo form
#   start_dt   --> start date in format datetime.datetime(yyyy, mm, dd) (Ex.: start_dt=datetime.datetime(2018, 1, 1) )
#   end_dt     --> end date in format datetime.datetime(yyyy, mm, dd)

    import pandas as pd
    from pandas_datareader import data as pdr
    import datetime
    import yfinance as yf

    data = pd.DataFrame()

    for i in range(len(asset_list)):
        while True:
                try:
                    d=yf.download(asset_list[i],start_dt,end_dt,threads=False)
                except:
                    print("Erro!!")
                else:
                    print(asset_list[i])
                    d=d['Adj Close']
                    n=asset_list[i]
                    data[n]=d
                    break

    return data