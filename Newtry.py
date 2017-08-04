import pandas as pd
import io
import requests
url="https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))

print(df.head)
