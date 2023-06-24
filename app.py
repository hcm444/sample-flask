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
import uuid  # Added import

ip_unique_ids = {}
nltk.download('punkt')
post_counts = {}
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

db_engine = create_engine('sqlite:///message_board.db')
Base = declarative_base()
Session = sessionmaker(bind=db_engine)


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    post_number = Column(Integer)
    timestamp = Column(DateTime)
    message = Column(String)
    referenced_post = Column(String(length=200))
    unique_id = Column(String)
    parent_post = Column(Integer)  # New column for reply threads


Base.metadata.create_all(db_engine)


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

    return ','.join(referenced_posts[:10])


POST_LIMIT_DURATION = timedelta(minutes=1)


def get_child_messages(messages, parent_id):
    # Recursive function to get child messages for a given parent ID
    child_messages = []
    for message in messages:
        if message['parent_post'] == parent_id:
            child_messages.append(message)
            child_messages.extend(get_child_messages(messages, message['post_number']))
    return child_messages


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
            'parent_post': message.parent_post  # Add parent_post field
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
    ip_address = request.remote_addr

    # Check if unique ID exists for the IP address
    if ip_address in ip_unique_ids:
        unique_id = ip_unique_ids[ip_address]
    else:
        # Generate a new unique ID for the IP address
        unique_id = str(uuid.uuid4())
        ip_unique_ids[ip_address] = unique_id

    # Extract referenced posts from the message using regular expressions
    references = extract_referenced_posts(message)

    # Get the parent post number from the referenced posts
    parent_post = references.split(',')[0] if references else None

    existing_message = session.query(Message).filter_by(message=message).first()
    if existing_message:
        session.close()
        return jsonify({'error': 'Error: This message already exists.'})

    total_posts = session.query(Message).count()
    if total_posts >= 100:
        most_recent_post = session.query(Message.post_number).order_by(Message.id.desc()).first()
        post_number = most_recent_post[0] + 1 if most_recent_post else 1
        oldest_posts = session.query(Message).order_by(Message.id).limit(total_posts - 99).all()
        for post in oldest_posts:
            session.delete(post)
    else:
        post_number = total_posts + 1

    timestamp = datetime.now()
    new_post = Message(
        post_number=post_number,
        timestamp=timestamp,
        message=message,
        referenced_post=references,
        unique_id=unique_id,
        parent_post=parent_post  # Store the parent post ID
    )
    session.add(new_post)
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
    app.run()
