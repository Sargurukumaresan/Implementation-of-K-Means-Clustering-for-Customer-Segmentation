# Ex08-Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
- Choose the number of clusters (K): 
  - Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

- Initialize cluster centroids: 
  -  Randomly select K data points from your dataset as the initial centroids of the clusters.

- Assign data points to clusters: 
  - Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

- Update cluster centroids: 
  - Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

- Repeat steps 3 and 4: 
  - Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

- Evaluate the clustering results: 
  - Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

- Select the best clustering solution: 
  - If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:

```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SARGURU K
RegisterNumber:  212222230134
```

```python
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:
### data.head():

![278949588-20e28c10-49ec-4912-9b52-aa1fa6046cdd](https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/3035bc29-d950-4dfc-9e00-1247d2d7b665)


### data.info():

![278949592-b72586a8-e2c9-46ab-bbbe-36120412beb3](https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/1fc0b2a0-2dce-482a-b082-388d77aa5a3c)


### NULL VALUES:

<img src="https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/e0c700d0-b94c-410b-b114-da575a0ab643">


### ELBOW GRAPH:
<img src="https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/d1d73971-8609-4293-a1a6-6381f6a06c29" width=50%>



### CLUSTER FORMATION:

<img src="https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/692d0fad-0f47-43bb-987e-f9813fd58bde">


### PREDICICTED VALUE:

<img src="https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/b3e35d8a-72e0-4f42-b27d-6cc5a7604a73">


### FINAL GRAPH(D/O):

<img src="https://github.com/Janarthanan2/ML_Ex08_Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393515/6d1710c9-d53f-41b5-a3cb-2bcb29100b51" width=35%>


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
