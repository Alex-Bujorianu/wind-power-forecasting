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
#Resampling to hours and taking the average of each hour
series = series.resample('H').mean()
print(series.shape)
print(series.head())

series = series.dropna(axis=0, how='any')
# Check for irregularity
for i in range(1, len(series)):
    irregular_count = 0
    if (series.index[i].hour - series.index[i-1].hour  > 1):
        print('Time series is irregular')
        print(series.index[i], series.index[i-1])
        break
print(series.shape)
print(series.head(10))
print(series.isnull().values.any())
# Make negative values 0
series.clip(lower=0, inplace=True)

if not series.isnull().values.any():
    series.to_csv("Cleaned_data.csv", index=True)
