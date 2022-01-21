import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python?select=Mall_Customers.csv

customers = pd.read_csv('Mall_Customers.csv')
customers_variables = customers.drop(['CustomerID', 'Gender', 'Age'], axis=1)
# print(customers.info())
# print(customers.describe())
# print(customers_variables.info())

# customers_norm = (customers_variables - customers_variables.min()) / (customers_variables.max() - customers_variables.min())
customers_norm = customers_variables
# print(customers_norm.describe())
# print(customers_norm)

plt.figure(figsize=(6, 6))

plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)',  data=customers_norm, s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) vs Annual Income (k$)')

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(customers_norm)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1, 11), wcss)
plt.title("Elbow method")
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")

clustering = KMeans(n_clusters=5, max_iter=300)
clustering.fit(customers_norm)

km1 = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=5, n_init=10, random_state=None, tol=0.0001, verbose=0)

km1.fit(customers_norm)
y = km1.predict(customers_norm)

customers_norm["label"] = y
customers["label"] = y

# print(customers)

plt.figure(figsize=(6, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue="label",
                palette=['green', 'orange', 'red', 'purple', 'blue'], legend='full',
                data=customers_norm, s=60)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) vs Annual Income (k$)')

# plt.axhline(50.0, linestyle='--')
# plt.axhline(51.5, linestyle='--')

# plt.axvline(38.5, linestyle='--')
# plt.axvline(68.5, linestyle='--')

plt.show()

customers.to_csv('customer_segmentation.csv')
