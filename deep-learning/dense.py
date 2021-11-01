import pandas as pd

##########
## Load ##
##########

df = pd.read_csv('file.csv')

y_label = 'class'
X = df.drop(y_label, axis=1)
y = df[y_label]

# one-hot encoder
from keras.utils import to_categorical
y = to_categorical(y)


#########
# Scale #
#########

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
	('scaler', StandardScaler())
])
X_scaled = pipe.fit_transform(X)


#########
# Split #
#########

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)


###########
# Network #
###########

from keras.models import Sequential
from keras.layers import Dense

num_features = 4
num_categories = 3

model = Sequential()
model.add(Dense(num_features*2, input_dim=num_features, activation='relu'))
model.add(Dense(num_features*2, input_dim=num_features, activation='relu'))
model.add(Dense(num_categories, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train,y_train,epochs=150, verbose=2)


###########
# Predict #
###########

y_pred_proba = model.predict(X_test)
y_pred = model.predict_classes(X_test)


###########
# Metrics #
###########

model.metrics_names
model.evaluate(x=X_test,y=y_test)

from sklearn.metrics import confusion_matrix,classification_report

confusion_matrix(y_test.argmax(axis=1),predictions)
print(classification_report(y_test.argmax(axis=1),predictions))


#############
# Save/Load #
#############

model.save('sample.h5')
from keras.models import load_model
model_v2 = load_model('sample.h5')
model_v2.predict_classes(X_test)