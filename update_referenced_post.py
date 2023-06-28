from app import Session, Message


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
