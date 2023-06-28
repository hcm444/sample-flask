import hashlib


def generate_unique_id(ip_address):
    # Create a hash object using the SHA-1 hash function
    hasher = hashlib.sha1()

    # Update the hash object with the IP address
    hasher.update(ip_address.encode('utf-8'))

    # Get the hexadecimal representation of the hash digest
    unique_id = hasher.hexdigest()

    # Return the unique ID
    return unique_id
