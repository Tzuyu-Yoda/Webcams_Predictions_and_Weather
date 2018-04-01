
# coding: utf-8

# In[12]:

import sys
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions, types


# In[13]:

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
# assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
# assert spark.version >= '2.1' # make sure we have Spark 2.1+
np.set_printoptions(threshold = np.nan)


# In[14]:

# in_directory = 'yvr-weather'
in_directory = sys.argv[1]
spark_data = spark.read.text(in_directory)
df = spark_data.toPandas()


# In[15]:

df1 = df.iloc[16:].reset_index(drop=True)
df2 = df1['value'].str.split(',', expand=True)
df2 = df2.apply(lambda s:s.str.replace('"', ""))
df2.columns = df2.iloc[0]
df3 = df2[df2['Temp (°C)'].notnull()].reset_index(drop=True)
df3 = df3[df3['Stn Press Flag'] != 'Stn Press Flag'].reset_index(drop=True)
df4 = df3.iloc[:, [0,1,2,3,4,6,8,10,12,14,16,18,24]]
df4 = df4[df4['Wind Dir (10s deg)'] != '']
df4 = df4[df4['Stn Press (kPa)'] != '']
df4.iloc[:, 1:4] = df4.iloc[:, 1:4].astype(float)
df4[['Stn Press (kPa)', 'Wind Spd (km/h)', 'Visibility (km)', 'Wind Dir (10s deg)', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)']] = df4[['Stn Press (kPa)', 'Wind Spd (km/h)', 'Visibility (km)', 'Wind Dir (10s deg)', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)']].astype(float)
df4 = df4[~df4.Weather.isin(['NA', 'Thunderstorms', 'Ice Pellets'])].reset_index(drop=True)
df4 = df4.sort_values(['Year', 'Month', 'Day', 'Time']).reset_index(drop=True)


# In[16]:

df4 = df4.rename(columns={'Stn Press (kPa)':'Stn Press', 'Wind Spd (km/h)':'Wind Spd', 'Visibility (km)':'Visibility', 'Wind Dir (10s deg)':'Wind Dir', 'Temp (°C)':'Temp', 'Dew Point Temp (°C)':'Dew Point Temp', 'Rel Hum (%)':'Rel Hum'})
df4.to_csv('weather_pd.csv', index=False)

