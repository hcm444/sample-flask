from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime, timedelta
import re
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_caching import Cache

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
    referenced_post = Column(String)

Base.metadata.create_all(db_engine)

def update_referenced_post(referenced_post, post_number):
    if referenced_post:
        session = Session()
        message = session.query(Message).filter_by(post_number=int(referenced_post)).first()
        if message:
            referenced_posts = message.referenced_post.split(',') if message.referenced_post else []
            if str(post_number) not in referenced_posts:
                referenced_posts.append(str(post_number))
                message.referenced_post = ','.join(referenced_posts)
        session.commit()
        session.close()

def extract_referenced_posts(message):
    referenced_posts = re.findall(r'>>(\d+)', message)
    return ','.join(referenced_posts)

POST_LIMIT_DURATION = timedelta(minutes=1)

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
            'referenced_by': message.referenced_post.split(',') if message.referenced_post else None
        }
        for message in messages
    ]
    session.close()
    return render_template('index.html', messages=messages_dict)

@app.route('/post', methods=['POST'])
def post():
    session = Session()
    message = request.form['message']
    ip_address = request.remote_addr

    if ip_address in post_counts and post_counts[ip_address]['count'] >= 3:
        if datetime.now() - post_counts[ip_address]['timestamp'] <= POST_LIMIT_DURATION:
            session.close()
            remaining_time = (POST_LIMIT_DURATION - (datetime.now() - post_counts[ip_address]['timestamp'])).seconds
            return jsonify({'error': 'Exceeded post limit. Please wait for {} seconds before posting again.'.format(remaining_time)})

        post_counts[ip_address]['count'] = 0

    if len(message) > 300:
        session.close()
        return jsonify({'error': 'Error: Message exceeds 300 characters.'})

    references = extract_referenced_posts(message)

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
    new_post = Message(post_number=post_number, timestamp=timestamp, message=message, referenced_post=references)
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
