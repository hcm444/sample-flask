# Description: This file contains the code for the Flask application that runs the message board.
from flask import Flask, render_template, request, redirect, jsonify

from datetime import datetime, timedelta
import re
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_caching import Cache
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

import hashlib

nltk.download('punkt')
post_counts = {}
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

db_engine = create_engine('sqlite:///message_board.db')
Base = declarative_base()
Session = sessionmaker(bind=db_engine)
Base.metadata.create_all(db_engine)
POST_LIMIT_DURATION = timedelta(minutes=1)

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_number = Column(Integer)
    timestamp = Column(DateTime)
    message = Column(String)
    referenced_post = Column(String(length=200))
    unique_id = Column(String)
    parent_post = Column(Integer)





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


def find_least_original_user():
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
        # Sort the user originalities in ascending order and return the least original user
        user_originalities.sort(key=lambda x: x[1])
        return user_originalities[0][0], user_originalities[0][1]

    return None, None


def generate_unique_id(ip_address):
    # Convert the IP address to bytes
    ip_bytes = ip_address.encode('utf-8')

    # Create a hash object using a cryptographic hash function (e.g., SHA256)
    hasher = hashlib.sha256()

    # Update the hash object with the IP address bytes
    hasher.update(ip_bytes)

    # Get the hexadecimal representation of the hash digest
    hash_code = hasher.hexdigest()

    # Return the first 16 characters as the unique ID
    unique_id = hash_code[:16]

    return unique_id


def calculate_sentiment(text):
    sentiment = sia.polarity_scores(text)['compound']
    return sentiment


def calculate_originality(new_post, existing_posts):
    # Combine new post and existing posts
    all_posts = existing_posts + [new_post]

    # Tokenize posts into sentences
    tokenized_posts = [nltk.sent_tokenize(post) for post in all_posts]

    # Flatten the list of sentences
    flattened_posts = [sentence for post in tokenized_posts for sentence in post]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(flattened_posts)

    # Calculate the cosine similarity between the new post and existing posts
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Calculate the average similarity
    average_similarity = similarity_matrix.mean()

    # Calculate the originality score
    originality_score = 1 - average_similarity

    return originality_score


def update_referenced_post(referenced_post, post_number):
    if referenced_post:
        session = Session()
        message = session.query(Message).filter_by(post_number=int(referenced_post)).first()
        if message:
            referenced_posts = message.referenced_post.split(',') if message.referenced_post else []
            if str(post_number) not in referenced_posts:
                # Limit the number of referenced posts to 10
                referenced_posts.append(str(post_number))
                referenced_posts = referenced_posts[-10:]
                message.referenced_post = ','.join(referenced_posts)
        session.commit()
        session.close()


def extract_referenced_posts(message):
    referenced_posts = re.findall(r'>>(\d{1,14})', message)

    session = Session()
    existing_post_numbers = [str(post[0]) for post in session.query(Message.post_number).all()]
    session.close()

    # Filter out referenced post numbers that don't exist in the database
    valid_referenced_posts = [post for post in referenced_posts if post in existing_post_numbers]

    return ','.join(valid_referenced_posts[:10])





def get_child_messages(messages, parent_id):
    # Recursive function to get child messages for a given parent ID
    child_messages = []
    for message in messages:
        if message['parent_post'] == parent_id:
            child_messages.append(message)
            child_messages.extend(get_child_messages(messages, message['post_number']))
    return child_messages


# app.py

@app.route('/chart')
@cache.cached(timeout=60)
def chart():
    session = Session()
    messages = session.query(Message).all()
    session.close()

    users = {}  # Dictionary to store user data

    # Process each message to calculate user-wise originality and sentiment
    for message in messages:
        user = message.unique_id

        if user not in users:
            users[user] = {'originality': [], 'sentiment': []}

        originality = float(calculate_originality(message.message, [m.message for m in messages]))
        sentiment = calculate_sentiment(message.message)

        users[user]['originality'].append(originality)
        users[user]['sentiment'].append(sentiment)

    # Prepare data for charting
    chart_data = []
    for user, data in users.items():
        avg_originality = sum(data['originality']) / len(data['originality'])
        avg_sentiment = sum(data['sentiment']) / len(data['sentiment'])
        chart_data.append({'user': user, 'originality': avg_originality, 'sentiment': avg_sentiment})

    return render_template('chart.html', data=chart_data)


@app.route('/')
@cache.cached(timeout=60)
def home():
    session = Session()
    messages = session.query(Message).all()
    messages_dict = [
        {
            'post_number': message.post_number,
            'timestamp': message.timestamp,
            'message': message.message,
            'referenced_by': message.referenced_post.split(',') if message.referenced_post else None,
            'originality': "{:.5f}".format(calculate_originality(message.message, [m.message for m in messages])),
            'unique_id': message.unique_id,
            'parent_post': message.parent_post,  # Add parent_post field
            'sentiment': calculate_sentiment(message.message)  # Use sentiment labels

        }
        for message in messages
    ]

    # Build a hierarchical structure of messages based on parent-child relationship
    root_messages = [message for message in messages_dict if message['parent_post'] is None]
    threaded_messages = []
    for root_message in root_messages:
        root_message['replies'] = get_child_messages(messages_dict, root_message['post_number'])
        threaded_messages.append(root_message)

    session.close()
    return render_template('index.html', messages=threaded_messages)


@app.route('/post', methods=['POST'])
def post():
    session = Session()
    message = request.form['message']
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    unique_id = generate_unique_id(ip_address)
    references = extract_referenced_posts(message)
    parent_post = references.split(',')[0] if references else None

    if len(message) > 500:
        session.close()
        return jsonify({'error': 'Error: Message should not exceed 500 characters.'})

    existing_message = session.query(Message).filter_by(message=message).first()
    if existing_message:
        session.close()
        return jsonify({'error': 'Error: This message already exists.'})

    if ip_address in post_counts:
        count = post_counts[ip_address]['count']
        timestamp = post_counts[ip_address]['timestamp']
        time_diff = datetime.now() - timestamp

        # Reset the post count if more than a minute has passed
        if time_diff > POST_LIMIT_DURATION:
            post_counts[ip_address] = {'count': 1, 'timestamp': datetime.now()}
        elif count >= 3:
            session.close()
            return jsonify({'error': 'Error: You can only post three times per minute.'})
        else:
            post_counts[ip_address]['count'] += 1
            post_counts[ip_address]['timestamp'] = datetime.now()
    else:
        post_counts[ip_address] = {'count': 1, 'timestamp': datetime.now()}

    total_posts = session.query(Message).count()
    if total_posts >= 500:
        most_recent_post = session.query(Message.post_number).order_by(Message.id.desc()).first()
        post_number = most_recent_post[0] + 1 if most_recent_post else 1
        oldest_posts = session.query(Message).order_by(Message.id).limit(total_posts - 499).all()
        for post in oldest_posts:
            session.delete(post)
    else:
        post_number = total_posts + 1

    timestamp = datetime.now()

    # Check if the message contains the special command for most original or least original users
    if '>>most_original' in message:
        most_original_user, most_original_score = find_most_original_user()
        if most_original_user is not None and most_original_score is not None:
            most_original_message = f"The most original user is {most_original_user} with an originality score of {most_original_score:.5f}"
            message += '\n\n' + most_original_message

    if '>>least_original' in message:
        least_original_user, least_original_score = find_least_original_user()
        if least_original_user is not None and least_original_score is not None:
            least_original_message = f"The least original user is {least_original_user} with an originality score of {least_original_score:.5f}"
            message += '\n\n' + least_original_message

    # Use a parameterized query to insert the new post
    query = text(
        "INSERT INTO messages (post_number, timestamp, message, referenced_post, unique_id, parent_post) "
        "VALUES (:post_number, :timestamp, :message, :referenced_post, :unique_id, :parent_post)"
    )
    params = {
        'post_number': post_number,
        'timestamp': timestamp,
        'message': message,
        'referenced_post': references,
        'unique_id': unique_id,
        'parent_post': parent_post
    }
    session.execute(query, params)
    session.commit()

    for referenced_post in references.split(','):
        update_referenced_post(referenced_post, post_number)

    if ip_address not in post_counts:
        post_counts[ip_address] = {'count': 1, 'timestamp': datetime.now()}
    else:
        post_counts[ip_address]['count'] += 1
        post_counts[ip_address]['timestamp'] = datetime.now()

    session.close()
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
