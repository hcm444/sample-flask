from app import sia


def calculate_sentiment(text):
    sentiment = sia.polarity_scores(text)['compound']
    return sentiment
