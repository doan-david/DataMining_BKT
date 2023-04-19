import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
df = pd.read_csv("D:\\filebaitap\\Khaiphadulieu\\test\\Mall_Customers.csv")
print(df.head())
# Visualize the distribution of Age using a histogram
sns.histplot(data=df, x='Age', bins=20)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
# Visualize the relationship between Annual Income and Spending Score using a scatter plot
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.title('Relationship between Annual Income and Spending Score')
plt.show()

X = df.iloc[:,[3,4]].values
# Using the elbow method to find the optimal number of clusters wcss = [] for i in range(1, 11): 
from sklearn.cluster import KMeans
# wcss = [] 
# for i in range(1, 11): 
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X) 
#     wcss.append(kmeans.inertia_)

# y_kmeans = kmeans.fit_predict(X)

# plt.ylabel('WCSS') 
# plt.show()

# 
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i , init='k-means++' , random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# 
kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)')
plt.legend() 

plt.show()