import json
import pandas as pd
from GetData import GetDataYahoo as gdy
import datetime
from datetime import date
import boto3
import io
from io import StringIO
import openpyxl
import yfinance as yf

def lambda_handler(event, context):
    # Get tickers
    s3_client = boto3.client('s3')
    s3_clientobj = s3_client.get_object(Bucket='testpairs', Key='SecuritiesList.xlsx')
    data = s3_clientobj['Body'].read()
    data_io = io.BytesIO(data)
    d=pd.read_excel(data_io)
    stocks=pd.DataFrame(d,columns=['Bloomberg','Yahoo','Setor'])
    #stocks_sector=pd.DataFrame(d,columns=['Yahoo','Setor'])
    stocks=stocks['Yahoo']
    
    
    # Get Data
    todays_date = date.today() 
    start_dt=datetime.datetime(2018, 8, 21)
    end_dt=datetime.datetime(todays_date.year, todays_date.month, todays_date.day)
    sdata=gdy(stocks,start_dt,end_dt)

    #Reindexing data
    sdata.columns = sdata.columns.str.replace(r'\.SA$','')
    sdata = sdata.sort_index(ascending=False)
    
    #Market free risk reference (temporary)
    sdataRef = yf.download('^BVSP',start_dt,end_dt,threads=False)
    bvsp = sdataRef['Adj Close']
    bvsp = bvsp.sort_index(ascending=False)

    #Store Data in S3
    bucket = 'testpairs'
    csv_buffer = StringIO()
    sdata.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'SecuritiesHistoricalData.csv').put(Body=csv_buffer.getvalue())
    bvsp.to_csv(csv_buffer)
    s3_resource.Object(bucket, 'MarketReference.csv').put(Body=csv_buffer.getvalue())

    #End
    print("END !!!!!!!!")
    
    return {
        'statusCode': 200,
        'body': json.dumps("END!")
    }