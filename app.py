# Description: This file contains the code for the Flask application that runs the message board.

from flask import Flask, render_template, request, redirect, jsonify

from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_caching import Cache
import nltk
from sqlalchemy import text
from nltk.sentiment import SentimentIntensityAnalyzer
from calculate_originality import calculate_originality
from calculate_sentiment import calculate_sentiment
from extract_referenced_posts import extract_referenced_posts
from find_least_original_user import find_least_original_user
from find_least_sentimental_user import find_least_sentimental_user
from find_most_original_user import find_most_original_user
from find_most_sentimental_user import find_most_sentimental_user
from gemerate_fortune import generate_fortune
from generate_unique_id import generate_unique_id
from get_child_messages import get_child_messages
from update_referenced_post import update_referenced_post

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

nltk.download('punkt')
post_counts = {}
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

db_engine = create_engine('sqlite:///message_board.db')
Base = declarative_base()
Session = sessionmaker(bind=db_engine)


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_number = Column(Integer)
    timestamp = Column(DateTime)
    message = Column(String)
    referenced_post = Column(String(length=200))
    unique_id = Column(String)
    parent_post = Column(Integer)


Base.metadata.create_all(db_engine)

POST_LIMIT_DURATION = timedelta(minutes=1)


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
            most_original_message = f"{most_original_user} : {most_original_score:.5f}"
            message += '\n\n' + most_original_message

    if '>>least_original' in message:
        least_original_user, least_original_score = find_least_original_user()
        if least_original_user is not None and least_original_score is not None:
            least_original_message = f"{least_original_user} : {least_original_score:.5f}"
            message += '\n\n' + least_original_message

    if '>>most_sentimental' in message:
        most_sentimental_user, most_sentimental_score = find_most_sentimental_user()
        if most_sentimental_user is not None and most_sentimental_score is not None:
            most_sentimental_message = f"{most_sentimental_user} : {most_sentimental_score:.5f}"
            message += '\n\n' + most_sentimental_message

    if '>>least_sentimental' in message:
        least_sentimental_user, least_sentimental_score = find_least_sentimental_user()
        if least_sentimental_user is not None and least_sentimental_score is not None:
            least_sentimental_message = f"{least_sentimental_user} : {least_sentimental_score:.5f}"
            message += '\n\n' + least_sentimental_message

    if '>>fortune' in message:
        fortune = generate_fortune()
        fortune_message = f"{fortune}."
        message += '\n\n' + fortune_message

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