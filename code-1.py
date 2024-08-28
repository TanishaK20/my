# -*- coding: utf-8 -*-
"""
Student ID -  	<59289>
Name - 			<Kainat>
Campus - 		<Sydney>
Subject code - 	<ICT 603>
Assessment no - <Assessment 1>
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate 1000 random dates and times within a specific range
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_times = [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(1000)]

# Generate random customer IDs
customer_ids = ['C' + str(i).zfill(4) for i in range(1, 1001)]

# Generate random product IDs
product_ids = ['P' + str(i).zfill(3) for i in range(1, 101)]

# Generate random quantities sold
quantities_sold = np.random.randint(1, 10, size=1000)

# Generate random unit prices
unit_prices = np.random.uniform(1, 100, size=1000)

# Calculate total transaction amounts
total_transaction_amounts = quantities_sold * unit_prices

# Generate random store IDs
store_ids = ['S' + str(i).zfill(3) for i in range(1, 11)]

# Randomly assign store IDs to transactions
store_ids = [random.choice(store_ids) for _ in range(1000)]

# Create DataFrame
data = {
    'Date & Time': date_times,
    'Customer ID': random.choices(customer_ids, k=1000),
    'Product ID': random.choices(product_ids, k=1000),
    'Quantity Sold': quantities_sold,
    'Unit Price': unit_prices,
    'Total Transaction Amount': total_transaction_amounts,
    'Store ID': store_ids
}

df = pd.DataFrame(data)

# Convert 'Date & Time' column to datetime format and sort the DataFrame
df['Date & Time'] = pd.to_datetime(df['Date & Time'])
df = df.sort_values(by='Date & Time').reset_index(drop=True)

# Print first few rows of DataFrame
print(df.head())

###############################################################################
'''
Section 3 - Analysis
Section 3.a - Descriptive Analysis
Descriptive statistics were calculated to summarize the central tendency, dispersion, and distribution of 
numerical variables in the dataset.A frequency count and summary for the 
categorical variable 'Store ID' were also performed.
'''

# 1. Descriptive Statistics
descriptive_stats = df.describe()
print("Descriptive Statistics:")
print(descriptive_stats)

# 2. Frequency Count for Store IDs (Categorical Variable)
store_visits_count = df['Store ID'].value_counts()
print("\nFrequency Count for Store Visits:")
print(store_visits_count)
 
# 3. Summary for Store IDs (Categorical Variable)
summary_categorical = df['Store ID'].describe()
print("\nSummary for Store IDs:")
print(summary_categorical)

# 4. Summary for Numerical Variables
summary_numerical = df[['Quantity Sold', 'Unit Price', 'Total Transaction Amount']].describe()
print("\nSummary for Numerical Variables:")
print(summary_numerical)

###############################################################################
'''
Section 3.b - Correlation Analysis
Correlation Analysis between numerical variables to understand the relationships between them.
'''

# Correlation analysis
numerical_columns = df[['Quantity Sold', 'Unit Price', 'Total Transaction Amount']]
correlation_matrix = numerical_columns.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

###############################################################################
'''
Section 3.c - Hypothesis Testing
 Hypothesis testing to determine if the average transaction amount is greater than a specified value.
'''

from scipy import stats

# Null Hypothesis (H0): The average transaction amount is less than or equal to the specified value
# Alternative Hypothesis (H1): The average transaction amount is greater than the specified value
specified_value = 500
transaction_mean = df['Total Transaction Amount'].mean()

# Perform a one-sample t-test
t_statistic, p_value = stats.ttest_1samp(df['Total Transaction Amount'], popmean=specified_value)

print("\nHypothesis Testing:")
print(f"Mean of Total Transaction Amount: {transaction_mean:.2f}")
print(f"T-Statistic: {t_statistic:.4f}, P-Value: {p_value:.4f}")

if p_value < 0.05 and t_statistic > 0:
    print(f"Reject the null hypothesis. The average transaction amount is significantly greater than ${specified_value}.")
else:
    print(f"Fail to reject the null hypothesis. The average transaction amount is not greater than ${specified_value}.")

###############################################################################
'''
Section 3.d - Time-Series Analysis
Since timestamps were included in the dataset, a time-series analysis was conducted to examine trends in sales over time.
'''

import matplotlib.pyplot as plt

# Time-series analysis: Plotting total sales over time
df.set_index('Date & Time', inplace=True)
time_series = df['Total Transaction Amount'].resample('M').sum()

plt.figure(figsize=(10, 6))
plt.plot(time_series, marker='o')
plt.title('Monthly Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

###############################################################################
'''
Section 4 - Visualization
Section 4.a - Total Sales per Store
A bar chart was created to visualize the total sales per store.
'''

import seaborn as sns

# Bar chart for sales per store
plt.figure(figsize=(10, 6))
sns.barplot(x='Store ID', y='Total Transaction Amount', data=df, estimator=sum)
plt.title('Total Sales per Store')
plt.show()

###############################################################################
'''
Section 4.b - Correlation Matrix
A heatmap was used to visualize the correlation between numerical variables.
'''

# Heatmap for correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
