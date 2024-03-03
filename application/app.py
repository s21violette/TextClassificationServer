from flask import Flask
from classificator import SentimentClassificator

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "<h1>Hello there!</h1>"
