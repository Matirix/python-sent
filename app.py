from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    return ss

def determine_sentiment(sentiment_scores):
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@app.route('/', methods=['GET', 'POST'])
def index():
    sentence = ''
    overall_sentiment = ''

    if request.method == 'POST':
        sentence = request.form['sentence']
        sentiment_scores = analyze_sentiment(sentence)
        overall_sentiment = determine_sentiment(sentiment_scores)

    return render_template('index.html', sentence=sentence, sentiment_scores=sentiment_scores, overall_sentiment=overall_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
