from app import Session, Message
from calculate_sentiment import calculate_sentiment


def calculate_user_sentiment(user_id):
    session = Session()

    # Get all messages posted by the user
    user_messages = session.query(Message).filter_by(unique_id=user_id).all()

    if not user_messages:
        session.close()
        return None

    # Calculate the sentiment score for each message
    sentiment_scores = []
    for message in user_messages:
        sentiment_scores.append(calculate_sentiment(message.message))

    session.close()

    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return average_sentiment

    return None
