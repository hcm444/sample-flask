# Description: This file contains the code for the Flask application that runs the message board.
import webbrowser


session_data = {}

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
import hashlib
import random

nltk.download('vader_lexicon')
nltk.download('punkt')

sia = SentimentIntensityAnalyzer()

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
    parent_post = Column(Integer)


Base.metadata.create_all(db_engine)


def generate_fortune():
    fortunes = [
        "All signs point to yes.",
        "Don't count on it.",
        "Outlook not so good.",
        "You may rely on it.",
        "Better not tell you now.",
        "Reply hazy, try again.",
        "It is certain.",
        "Cannot predict now.",
        "Yes, definitely.",
        "My sources say no.",
        "Signs point to yes.",
        "Ask again later.",
        "Very doubtful.",
        "Most likely.",
        "It is decidedly so.",
        "Without a doubt.",
        "Yes, definitely.",
        "My reply is no.",
        "Outlook good.",
        "Concentrate and ask again."
    ]
    return random.choice(fortunes)







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


POST_LIMIT_DURATION = timedelta(minutes=1)


def get_child_messages(messages, parent_id):
    # Recursive function to get child messages for a given parent ID
    child_messages = []
    for message in messages:
        if message['parent_post'] == parent_id:
            child_messages.append(message)
            child_messages.extend(get_child_messages(messages, message['post_number']))
    return child_messages


# app.py
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', error='404 - Page not found'), 404

@app.route('/snake')
def snake():
    return render_template('snake.html')

@app.route("/save_high_score", methods=["POST"])
def save_high_score():
    high_score = request.json.get("score")
    ip_address = request.remote_addr

    print(ip_address, high_score)

    # Update the highest score and unique ID if necessary
    if high_score > session_data.get("highest_score", 0):
        session_data["highest_score"] = high_score

    return jsonify({"highest_score": session_data.get("highest_score"), "winner_id": session_data.get("winner_id")})




@app.route('/')
@cache.cached(timeout=60)
def home():
    session = Session()
    highest_score = session_data.get("highest_score")
    winner_id = session_data.get("winner_id")


    messages = session.query(Message).all()
    messages_dict = [
        {
            'post_number': message.post_number,
            'timestamp': message.timestamp,
            'message': message.message,
            'referenced_by': message.referenced_post.split(',') if message.referenced_post else None,
            'originality': "{:.5f}".format(calculate_originality(message.message, [m.message for m in messages])),
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
    return render_template('index.html', messages=threaded_messages, highest_score=highest_score, winner_id=winner_id)

@app.route('/post', methods=['POST'])
def post():
    session = Session()
    message = request.form['message']
    ip_address = request.remote_addr

    references = extract_referenced_posts(message)
    parent_post = references.split(',')[0] if references else None

    if len(message) > 500:
        session.close()
        return jsonify({'error': 'Error: Message should not exceed 500 characters.'})

    if len(message) == 0:
        session.close()
        return jsonify({'error': 'Error: Message should not be 0 characters.'})

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

    if '>>fortune' in message:
        fortune = generate_fortune()
        fortune_message = f"{fortune}."
        message += '\n\n' + fortune_message

    # Use a parameterized query to insert the new post
    query = text(
        "INSERT INTO messages (post_number, timestamp, message, referenced_post, parent_post) "
        "VALUES (:post_number, :timestamp, :message, :referenced_post, :parent_post)"
    )

    print(ip_address)
    params = {
        'post_number': post_number,
        'timestamp': timestamp,
        'message': message,
        'referenced_post': references,
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