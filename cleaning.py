import pandas as pd

series = pd.read_csv("Turbine_Data.csv")
# Remove uninteresting columns
series = series[["Unnamed: 0", "ActivePower", "WindSpeed", "WindDirection"]]
print(series.shape)

#convert Datetime to pandas datetimeand set it as index
series["Unnamed: 0"] = pd.to_datetime(series["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S%z")
datetime_index = pd.DatetimeIndex(series["Unnamed: 0"].values)
series = series.set_index(datetime_index)
#Drop redundant column
series = series.drop(['Unnamed: 0'], axis=1)
print(series.shape)
#Resampling to hoursand taking the average of each hour
series = series.resample('H').mean()
print(series.shape)

series = series.dropna(axis=0, how='any')
print(series.shape)
print(series.head(10))
print(series.isnull().values.any())
#fill in missing values with zero using the fillforward function
series = series.fillna(0).astype(float)

series.to_csv("Cleaned_data.csv", index=True)
