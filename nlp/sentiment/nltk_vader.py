# download vader
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# define the main function
def get_sentiment_score(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    score = sentiment_analyzer.polarity_scores(text)['compound']
    verdict = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
    return verdict, score