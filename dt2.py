from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv('customer_segmentation.csv')
df = df.drop(["CustomerID", "Gender", "Age"], axis=1)
df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)

X = df
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
clf.fit(X_train, y_train)

print(df)

tree.plot_tree(clf, feature_names=["Annual Income", "Spending Score"])
plt.show()
