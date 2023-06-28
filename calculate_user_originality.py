from app import Session, Message
from calculate_originality import calculate_originality


def calculate_user_originality(user_id):
    session = Session()

    # Get all messages posted by the user
    user_messages = session.query(Message).filter_by(unique_id=user_id).all()

    if not user_messages:
        session.close()
        return None

    # Calculate the originality score for each message
    originality_scores = []
    for message in user_messages:
        existing_messages = [m.message for m in session.query(Message).all() if m != message]
        originality_scores.append(calculate_originality(message.message, existing_messages))

    session.close()

    if originality_scores:
        average_originality = sum(originality_scores) / len(originality_scores)
        return average_originality

    return None
