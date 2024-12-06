import pandas as pd

# Load the hourly data
data_hourly = pd.read_csv('C:\Users\M Swarna\Desktop\ML\hour.csv')

# Load the daily data
data_daily = pd.read_csv('C:\Users\M Swarna\Desktop\ML\day.csv')

# Display the first few rows
print(data_hourly.head())
print(data_daily.head())
