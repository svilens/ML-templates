# NMF (non-negative matrix factorization)

import pandas as pd

df = pd.read_csv('text.csv')
len(df)
df.head()

# create doc-term matrix using count vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df)
dtm.shape

# fit the NMF instance on the doc-term matrix
from sklearn.decomposition import NMF
model = NMF(n_components=7, random_state=42)
model.fit(dtm)

len(vectorizer.get_feature_names())

# print out the top words for each topic
for index,topic in enumerate(model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# create a label for each topic based on the top words
topic_labels_dict = {0:'topic1', 1:'topic2', 2:'topic3'}

# transform the doc-term matrix into score by topic
topic_results = model.transform(dtm)

# get the topic index with highest score
df['topic_num'] = topic_results.argmax(axis=1)

# apply topic labels
df['topic_label'] = df['topic_num'].map(topic_labels_dict)
df['topic_label'].value_counts()