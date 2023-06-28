from app import Session, Message
from calculate_user_originality import calculate_user_originality


def find_most_original_user():
    session = Session()

    # Get all unique user IDs
    unique_user_ids = session.query(Message.unique_id.distinct()).all()

    user_originalities = []
    for user_id in unique_user_ids:
        originality = calculate_user_originality(user_id[0])
        if originality is not None:
            user_originalities.append((user_id[0], originality))

    session.close()

    if user_originalities:
        # Sort the user originalities in descending order and return the most original user
        user_originalities.sort(key=lambda x: x[1], reverse=True)
        return user_originalities[0][0], user_originalities[0][1]

    return None, None
