# Description: This file contains the code for the Flask application that runs the message board.
import webbrowser

session_data = {}

from flask import Flask, render_template, request, redirect, jsonify, flash

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
from flask_login import UserMixin, LoginManager, login_required, login_user, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, EqualTo

nltk.download('vader_lexicon')
nltk.download('punkt')

sia = SentimentIntensityAnalyzer()

post_counts = {}
app = Flask(__name__)
app.config['SECRET_KEY'] = 'bigpeen'

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

db_engine = create_engine('sqlite:///message_board.db')
Base = declarative_base()
Session = sessionmaker(bind=db_engine)


class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(64), index=True, unique=True)
    email = Column(String(120), unique=True, index=True)
    password_hash = Column(String(128))
    Session = sessionmaker(bind=db_engine)
    db_session = Session()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def query():
        Session = sessionmaker(bind=db_engine)
        db_session = Session()
        return db_session.query(User)


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    post_number = Column(Integer)
    timestamp = Column(DateTime)
    message = Column(String)
    referenced_post = Column(String(length=200))
    unique_id = Column(String)
    parent_post = Column(Integer)


login_manager = LoginManager()
login_manager.init_app(app)

Base.metadata.create_all(db_engine)


@login_manager.user_loader
def load_user(user_id):
    Session = sessionmaker(bind=db_engine)
    db_session = Session()
    return db_session.query(User).get(int(user_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        Session = sessionmaker(bind=db_engine)
        db_session = Session()
        db_session.add(user)
        db_session.commit()
        flash('Registration successful. Please log in.')
        return redirect('/login')
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        Session = sessionmaker(bind=db_engine)
        db_session = Session()
        user = db_session.query(User).filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page or '/')
        else:
            flash('Invalid email or password.')
    return render_template('login.html', title='Login', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/index')


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
    # Create a hash object using the SHA-1 hash function
    hasher = hashlib.sha1()

    # Update the hash object with the IP address
    hasher.update(ip_address.encode('utf-8'))

    # Get the hexadecimal representation of the hash digest
    unique_id = hasher.hexdigest()

    # Take the first 8 characters of the unique ID
    truncated_id = unique_id[:8]

    # Select a random oceanic animal
    animal = "prawn"

    # Concatenate the truncated ID with the oceanic animal string
    final_id = animal + truncated_id

    # Return the final ID
    return final_id


def calculate_sentiment(text):
    sentiment = sia.polarity_scores(text)['compound']
    return sentiment


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


def find_least_sentimental_user():
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
        # Sort the user sentiments in ascending order and return the least sentimental user
        user_sentiments.sort(key=lambda x: x[1])
        return user_sentiments[0][0], user_sentiments[0][1]

    return None, None


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
    winner_id = generate_unique_id(ip_address)
    print(ip_address, winner_id, high_score)

    # Update the highest score and unique ID if necessary
    if high_score > session_data.get("highest_score", 0):
        session_data["highest_score"] = high_score
        session_data["winner_id"] = winner_id

    return jsonify({"highest_score": session_data.get("highest_score"), "winner_id": session_data.get("winner_id")})


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
def home():
    if current_user.is_authenticated:
        # User is logged in, show the message board
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
                'unique_id': message.unique_id,
                'parent_post': message.parent_post,
                'sentiment': calculate_sentiment(message.message)
            }
            for message in messages
        ]

        root_messages = [message for message in messages_dict if message['parent_post'] is None]
        threaded_messages = []
        for root_message in root_messages:
            root_message['replies'] = get_child_messages(messages_dict, root_message['post_number'])
            threaded_messages.append(root_message)

        session.close()
        return render_template('index.html', messages=threaded_messages, highest_score=highest_score,
                               winner_id=winner_id)
    else:
        # User is not logged in, redirect to the login page
        return redirect('/login')


@app.route('/post', methods=['POST'])
@login_required
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
    unique_id = generate_unique_id(ip_address)
    print(ip_address, unique_id)
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
