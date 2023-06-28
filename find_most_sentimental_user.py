from app import Session, Message
from calculate_user_sentiment import calculate_user_sentiment


def find_most_sentimental_user():
    session = Session()

    # Get all unique user IDs
    unique_user_ids = session.query(Message.unique_id.distinct()).all()

    user_sentiments = []
    for user_id in unique_user_ids:
        sentiment = calculate_user_sentiment(user_id[0])
        if sentiment is not None:
            user_sentiments.append((user_id[0], sentiment))

    session.close()

    if user_sentiments:
        # Sort the user sentiments in descending order and return the most sentimental user
        user_sentiments.sort(key=lambda x: x[1], reverse=True)
        return user_sentiments[0][0], user_sentiments[0][1]

    return None, None
