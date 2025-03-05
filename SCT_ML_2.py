import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\Mall_Customers.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Perform some basic analysis
average_age = df['Age'].mean()
average_income = df['Annual Income (k$)'].mean()
average_spending_score = df['Spending Score (1-100)'].mean()

print()
print(f'Average Age: {average_age}')
print(f'Average Annual Income (k$): {average_income}')
print(f'Average Spending Score (1-100): {average_spending_score}')

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply k-means clustering with Annual Income and Spending Score
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save the clustered data to a new CSV file
df.to_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\Mall_Customers.csv', index=False)

# Plot the clusters
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis') #c=df['Cluster'] : Colors the points based on the cluster each customer belongs to.
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()

# Apply k-means clustering with  Customer ID and Spending Score
X = df[['CustomerID', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save the clustered data to a new CSV file
df.to_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\Mall_Customers.csv', index=False)

# Plot the clusters
plt.scatter(df['CustomerID'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis') #c=df['Cluster'] : Colors the points based on the cluster each customer belongs to.
plt.xlabel('Customer ID')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()

# Apply k-means clustering with  Customer ID and Annual Income
X = df[['CustomerID', 'Annual Income (k$)']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save the clustered data to a new CSV file
df.to_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\Mall_Customers.csv', index=False)

# Plot the clusters
plt.scatter(df['CustomerID'], df['Annual Income (k$)'], c=df['Cluster'], cmap='viridis') #c=df['Cluster'] : Colors the points based on the cluster each customer belongs to.
plt.xlabel('Customer ID')
plt.ylabel('Annual Income (k$)')
plt.title('Customer Segments')
plt.show()