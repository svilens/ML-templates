import numpy as np
import pandas as pd
from datetime import datetime

###########
# PreProc #
###########

# load data
train = pd.read_csv('train.csv')
X_submission = pd.read_csv('test.csv')

y_label = 'claim'

# sampling
train_sample = train.sample(100000)
y = train_sample[y_label]
X = train_sample.iloc[:,:-1]

# fill missing values with mean
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    SimpleImputer(strategy="mean")
)
X = pipe.fit_transform(X.copy())

# split to train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=81
)


##########
## UMAP ##
##########
len(X_train)

import umap

manifold = umap.UMAP()
manifold.fit(X_train, y_train)
X_reduced = manifold.transform(X_train)

X_reduced.shape

import matplotlib.pyplot as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, s=0.5);


###########
## Scale ##
###########
from sklearn.preprocessing import QuantileTransformer

pipe = Pipeline(
    SimpleImputer(strategy="mean"),
    QuantileTransformer()
)
X_train = pipe.fit_transform(X_train.copy())

manifold = umap.UMAP()
manifold.fit(X_train, y_train)
X_reduced_scaled = manifold.transform(X_train)
plt.scatter(X_reduced_scaled[:, 0], X_reduced_scaled[:, 1], c=y_train, s=0.5);


##########
## Plot ##
##########
import umap.plot
umap.plot.points(manifold, labels=y_train, theme="fire");

n_neighbors = [10, 40, 100, 200]
fig, ax = plt.subplots(2, 2, figsize=(20, 20))
for ax, n in zip(ax.flatten(), n_neighbors):
    manifold = umap.UMAP(n_neighbors=n, random_state=1121218)
    manifold.fit(X_reduced_scaled, y_train)
    umap.plot.points(manifold, labels=y_train, ax=ax, theme="fire")
    ax.set_title(f"UMAP with n_neighbors={n}", fontsize="x-large")