import pandas as pd

df = pd.read_csv('text.csv')

###########
# Preproc #
###########

df.head()

# Check for NaN values
df.isna().sum()

# Check for whitespace strings
blanks = []

for row in df.itertuples():
    if type(row.review) == str:
        if str(row.review).isspace():
            blanks.append(row.index)

blanks

df.dropna(inplace=True)
df.drop(blanks, inplace=True)


#########
# Split #
#########

from sklearn.model_selection import train_test_split
label_x = 'review'
label_y = 'label'

X = df[label_x]
y = df[label_y]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=81
)

#############
# Vectorize #
#############

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

model.fit(X_train, y_train)


###########
# Predict #
###########

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)
print(classification_report(y_pred, y_test))