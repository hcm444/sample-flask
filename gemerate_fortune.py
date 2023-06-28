import random


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
