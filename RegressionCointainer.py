import numpy as np
from numpy import cumsum, polyfit, log, sqrt, subtract
import pandas as pd
import openpyxl
import statsmodels
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LinearRegression
import pytrends 
import quandl
from pandas_datareader import data as pdr
import datetime
import functools
import math
from hurst import compute_Hc, random_walk
from GetData import GetDataYahoo as gdy
from datetime import date
import io
from io import StringIO


#Hurst Exponent
def hurst_exponent(data):
    lags = range(2,100)
    tau = [sqrt(np.std(subtract(data[lag:], data[:-lag]))) for lag in lags]
    m = np.polyfit(log(lags), log(tau), 1)
    hurst = m[0]*2
    return hurst


# Get Data - Reads CSV file from S3 Bucket
s3_client = boto3.client('s3')
s3_clientobj = s3_client.get_object(Bucket='testpairs', Key='SecuritiesHistoricalData.csv')
data = s3_clientobj['Body'].read()
data_io = io.BytesIO(data)
sdata=pd.read_csv(data_io)
    
#Fill NA's with zeros
sdata.fillna(0,inplace=True)

print("\nBEGIN\n")

#Timestamp - remove after twiking
start_timestamp = datetime.datetime.now()

#Begin
n = sdata.shape[1]
CointPairs = pd.DataFrame()
corr_matrix = pd.DataFrame()

#Define Index represented by JOB INDEX (AWS)
#Security_Index = AWS_BATCH_JOB_ARRAY_INDEX
Security_Index = 0

#Defines Independent Variable for Linear Regression
Stock_Ind = sdata.iloc[:,Security_Index].to_numpy()                               #Gets first (Independent) stock
Stock_Ind_ = Stock_Ind.reshape(-1, 1)                                             # 1 column and N lines | 2 dimensions
size = Stock_Ind.size
    

#Perform Linear Regression for each Security
for j in range(n):    
    if (j != Security_Index):                                                    #To avoid same stocks in the pair
                
        #Defines Dependent Variable for Linear Regression
        Stock_Dep = sdata.iloc[:,j].to_numpy()                                    #Gets second (Dependent) stock
        Stock_Dep_ = Stock_Dep.reshape(-1, 1)                                     # 1 column and N lines | 2 dimensions
        
        #Multiple window size cointegration test
        for k in range(250, 50, -10):
            Stock_Ind_sliced = Stock_Ind_[(size-k):(size+1)]                      #Slicing test
            Stock_Dep_sliced = Stock_Dep_[(size-k):(size+1)]       

            #Linear Regression
            model = LinearRegression().fit(Stock_Ind_sliced, Stock_Dep_sliced)
            r_sq = model.score(Stock_Ind_sliced, Stock_Dep_sliced)                #R2 Linear Regression result
            coefLin = model.intercept_                                            #Intercept
            coefAng = model.coef_                                                 #Slope

            Residual = Stock_Dep_sliced - model.predict(Stock_Ind_sliced)         #Residual array

                    
            #Augmented Dickey-Fuller (ADF) Test
            #result = adfuller(Residual,maxlag=1,autolag=None)                    #Augmented Dickey-Fuller cointegration test
            result = adfuller(Residual)
            coint = (1 - result[1])*100

                    
            #Position, last closed price - Check if still necessary (??)
            std = np.std(Residual)
            pos_ = Stock_Dep_sliced.size
            error_ = (Stock_Dep_sliced[pos_-1] - (Stock_Ind_sliced[pos_-1]*coefAng + coefLin))/std

            #Correlation: Pearson
            corr_matrix = pd.DataFrame(Stock_Ind_sliced)
            corr_matrix['Dep'] = Stock_Dep_sliced
            correr_ = corr_matrix.corr(method='pearson')

            #Correlation: Pearson (for Returns)
            #corr_matrix = pd.DataFrame(Stock_Ind_sliced).pct_change()
            #corr_matrix['Dep'] = pd.DataFrame(Stock_Dep_sliced).pct_change()
            #correr_ = corr_matrix.corr(method='pearson')
            #corr_matrix.to_excel('Test.xlsx')

            #Half-life: Ornstein-Uhlenbeck
            res_ = Residual
            lagged = np.roll(res_, 1)
            delta = res_-lagged
            delta[0] = 1
            mod = LinearRegression().fit(lagged, delta)
            gamma = mod.coef_
            gamma_int = mod.intercept_
            half_life = -1*np.log(2)/gamma

            #Hurst Exponent
            if (k >= 100):
                H, c, data = compute_Hc(Residual, kind='change', simplified=True)   #Very time consumming...
                #H = hurst_exponent(Residual)
            else:
                H = 'NAN'

                    

            #Data Output - Building Dataframe
            CointPairsTemp = pd.DataFrame({'Dependente':[sdata.columns[j]], 'Independente':[sdata.columns[Security_Index]]})
            CointPairsTemp['Period'] = k
            CointPairsTemp['Coint'] = coint
            CointPairsTemp['STD'] = std
            CointPairsTemp['Last_Dev'] = error_
            CointPairsTemp['R2'] = r_sq
            CointPairsTemp['Slope'] = coefAng
            CointPairsTemp['Intercept'] = coefLin
            CointPairsTemp['Half-life'] = math.ceil(half_life)
            CointPairsTemp['Hurst'] = H
            CointPairsTemp['Correlation'] = correr_.iloc[0,1]
            CointPairs = CointPairs.append(CointPairsTemp)

#End

#Timestamp - Remove after twiking
total_timestamp = datetime.datetime.now() - start_timestamp
print(str(total_timestamp))

#Output data
CointPairs = CointPairs.replace(regex=r'\.SA$', value='')

#Store Data in S3
bucket = 'testpairs'
csv_buffer = StringIO()
CointPairs.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'SecurityRegression_' + str(Security_Index) + '.csv').put(Body=csv_buffer.getvalue())