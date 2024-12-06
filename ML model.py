import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the hourly data
data_hourly = pd.read_csv(r'C:\Users\M Swarna\Desktop\ML\hour.csv')

# Load the daily data
data_daily = pd.read_csv(r'C:\Users\M Swarna\Desktop\ML\day.csv')

# Display the first few rows
print("Hourly Data:")
print(data_hourly.head())
print("\nDaily Data:")
print(data_daily.head())

# Select relevant features for analysis
data_hourly = data_hourly[['temp', 'hum', 'windspeed', 'hr', 'season', 'holiday', 'workingday', 'cnt']]
data_hourly.dropna(inplace=True)

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

### 1. Average Bike Rentals by Hour (Line Plot) ###
plt.figure(figsize=(10, 6))
hourly_rentals = data_hourly.groupby('hr')['cnt'].mean()
plt.plot(hourly_rentals.index, hourly_rentals.values, marker='o', linestyle='-', color='#42a5f5', label='Average Rentals')
plt.title('Average Bike Rentals per Hour of the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Number of Bike Rentals')
plt.xticks(range(0, 24), labels=range(0, 24))
plt.grid(True)
plt.legend()
plt.annotate('Peak Rentals', xy=(17, hourly_rentals[17]), xytext=(18, hourly_rentals[17]+10),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
print("Average bike rentals plotted by hour.")

### 2. Heatmap of Correlation Among Features ###
corr_matrix = data_hourly.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='white')
plt.title('Correlation Heatmap of Bike Sharing Features')
plt.show()
print("Correlation heatmap displayed.")

### 3. Regression Plot for Rentals Based on Temperature ###
plt.figure(figsize=(8, 6))
sns.regplot(x='temp', y='cnt', data=data_hourly, scatter_kws={'s': 50, 'color': '#f57c00'}, line_kws={'color': 'blue'})
plt.title('Regression Plot: Bike Rentals vs Temperature')
plt.xlabel('Temperature (Normalized)')
plt.ylabel('Bike Rentals')
plt.grid(True)
plt.show()
print("Regression plot displayed for rentals based on temperature.")

### 4. Regression Plot for Rentals Based on Humidity ###
plt.figure(figsize=(8, 6))
sns.regplot(x='hum', y='cnt', data=data_hourly, scatter_kws={'s': 50, 'color': '#ff7043'}, line_kws={'color': 'green'})
plt.title('Regression Plot: Bike Rentals vs Humidity')
plt.xlabel('Humidity (Normalized)')
plt.ylabel('Bike Rentals')
plt.grid(True)
plt.show()
print("Regression plot displayed for rentals based on humidity.")

### 5. Regression Plot for Rentals Based on Windspeed ###
plt.figure(figsize=(8, 6))
sns.regplot(x='windspeed', y='cnt', data=data_hourly, scatter_kws={'s': 50, 'color': '#7e57c2'}, line_kws={'color': 'red'})
plt.title('Regression Plot: Bike Rentals vs Windspeed')
plt.xlabel('Windspeed (Normalized)')
plt.ylabel('Bike Rentals')
plt.grid(True)
plt.show()
print("Regression plot displayed for rentals based on windspeed.")

### 6. Analyze Bike Rentals for Different Seasons ###
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='cnt', data=data_hourly, hue='season', palette='coolwarm', dodge=False)
plt.title('Bike Rentals by Season')
plt.xlabel('Season (1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall)')
plt.ylabel('Number of Bike Rentals')
plt.legend([],[], frameon=False)  # Optional: to remove redundant legends
plt.show()
print("Boxplot for bike rentals by season displayed.")

### 7. Analyze Bike Rentals on Working Days vs Holidays ###
plt.figure(figsize=(10, 6))
sns.boxplot(x='workingday', y='cnt', data=data_hourly, hue='workingday', palette='Set2', dodge=False)
plt.title('Bike Rentals: Working Days vs Holidays')
plt.xlabel('Working Day (0 = Holiday, 1 = Working Day)')
plt.ylabel('Number of Bike Rentals')
plt.legend([],[], frameon=False)  # Optional: to remove redundant legends
plt.show()
print("Boxplot for bike rentals on working days vs holidays displayed.")

### 8. Visualizing Missing Values with a Creative Bar Plot ###
missing_values = data_hourly.isnull().sum()
missing_values = missing_values[missing_values > 0]

plt.figure(figsize=(10, 6))
colors = sns.color_palette("coolwarm", len(missing_values))
sns.barplot(x=missing_values.values, y=missing_values.index, palette=colors)

for i, value in enumerate(missing_values.values):
    plt.text(value, i, f' {value}', va='center', color='black', fontweight='bold')

plt.title('Missing Values in Hourly Data', fontsize=16, fontweight='bold')
plt.xlabel('Number of Missing Values', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()
print("Missing values barplot displayed.")

# Prepare data for predictive modeling
X = data_hourly[['temp', 'hum', 'windspeed', 'hr', 'season', 'holiday', 'workingday']]
y = data_hourly['cnt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
rmse_linear = mse_linear ** 0.5  # Calculate RMSE

print(f"Linear Regression - Mean Absolute Error: {mae_linear}")
print(f"Linear Regression - Mean Squared Error: {mse_linear}")
print(f"Linear Regression - Root Mean Squared Error: {rmse_linear}")
print(f"Linear Regression - R-squared Score: {r2_linear}")

# Plot predicted vs actual rentals for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, color='#8e24aa', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Bike Rentals (Linear Regression)')
plt.xlabel('Actual Rentals')
plt.ylabel('Predicted Rentals')
plt.grid(True)
plt.show()

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5  # Calculate RMSE

print(f"Random Forest - Mean Absolute Error: {mae_rf}")
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - Root Mean Squared Error: {rmse_rf}")
print(f"Random Forest - R-squared Score: {r2_rf}")

# Plot predicted vs actual rentals for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='#8e24aa', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Bike Rentals (Random Forest)')
plt.xlabel('Actual Rentals')
plt.ylabel('Predicted Rentals')
plt.grid(True)
plt.show()

print("Scatter plot for actual vs predicted rentals displayed.")

