import re

from app import Session, Message


def extract_referenced_posts(message):
    referenced_posts = re.findall(r'>>(\d{1,14})', message)

    session = Session()
    existing_post_numbers = [str(post[0]) for post in session.query(Message.post_number).all()]
    session.close()

    # Filter out referenced post numbers that don't exist in the database
    valid_referenced_posts = [post for post in referenced_posts if post in existing_post_numbers]

    return ','.join(valid_referenced_posts[:10])
