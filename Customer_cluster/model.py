#Preparing a cluster of customers to predict the purchase power based on their income and spending
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
df = pd.read_csv("Mall_Customers.csv")

#df.info()
#print(df.shape)

X = df[["Annual Income (k$)","Spending Score (1-100)"]]
#X.head()
#X.info()
wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=1)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

print(wcss_list)

#Visualize the results
# plt.plot(range(1,11),wcss_list)
# plt.title("Elbow Method Graph")
# plt.xlabel("Number of Clusters")
# plt.ylabel("WCSS List")
# plt.show()


model = KMeans(n_clusters=6,init="k-means++",random_state=1)
y_predict = model.fit_predict(X)

X_array=X.values
#Plotting the graph of clusters
plt.scatter(X_array[y_predict==0,0],X_array[y_predict==0,1], s=100,color="Green")
plt.scatter(X_array[y_predict==1,0],X_array[y_predict==1,1], s=100,color="Red")
plt.scatter(X_array[y_predict==2,0],X_array[y_predict==2,1], s=100,color="Yellow")
plt.scatter(X_array[y_predict==3,0],X_array[y_predict==3,1], s=100,color="Blue")
plt.scatter(X_array[y_predict==4,0],X_array[y_predict==4,1], s=100,color="Black")
plt.scatter(X_array[y_predict==5,0],X_array[y_predict==5,1], s=100,color="Pink")


plt.title("Clusters of our plot")
plt.show()

#Saving the model
joblib.dump(model,"Model.pkl")
print("Model saved")