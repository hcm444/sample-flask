def get_child_messages(messages, parent_id):
    # Recursive function to get child messages for a given parent ID
    child_messages = []
    for message in messages:
        if message['parent_post'] == parent_id:
            child_messages.append(message)
            child_messages.extend(get_child_messages(messages, message['post_number']))
    return child_messages
