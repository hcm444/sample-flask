from flask import Flask, render_template, request, redirect
from datetime import datetime
import re
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_caching import Cache



app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


# Configure the database connection
db_engine = create_engine('sqlite:///message_board.db')
Base = declarative_base()
Session = sessionmaker(bind=db_engine)

# Define the Message model
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

@app.route('/')
@cache.cached(timeout=60)  # Cache the result for 60 seconds
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

    if len(message) > 300:
        session.close()
        return "Error: Message exceeds 300 characters."

    references = extract_referenced_posts(message)


    post_number = session.query(Message).count() + 1
    timestamp = datetime.now()
    new_post = Message(post_number=post_number, timestamp=timestamp, message=message, referenced_post=references)
    session.add(new_post)
    session.commit()

    for referenced_post in references.split(','):
        update_referenced_post(referenced_post, post_number)

    session.close()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
