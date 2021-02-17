FROM python:3.8

ADD RegressionContainer.py .

RUN pip install numpy pandas openpyxl statsmodels sklearn pytrends quandl datetime functools math hurst io

CMD [ "python", "./RegressionContainer.py" ]