import pandas as pd

# load the training dataset
data = pd.read_csv('data.csv')
y_label = 'group'

features = data.drop(y_label, axis=1)


#########################
# Optimal n of clusters #
#########################

# we can define the optimal number of clusters using the elbow method
# by calculating the within cluster sum of squares (WCSS)
# which is an aggregated measure of clusters' tightness

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

# Create 10 models with 1 to 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    # Fit the data points
    kmeans.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmeans.inertia_)
    
#Plot the WCSS values onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


n_clusters_optim = 3


###########
# K-Means #
###########

from sklearn.cluster import KMeans

model = KMeans(n_clusters=n_clusters_optim, init='k-means++', n_init=100, max_iter=1000)
km_clusters = model.fit_predict(features.values)

# we can use Principal Component Analysis (PCA) for visualization purposes
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaled_features = MinMaxScaler().fit_transform(features)
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)


def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)