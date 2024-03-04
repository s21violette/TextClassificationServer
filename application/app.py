from flask import Flask, request, abort, redirect, url_for, render_template
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from wtforms import StringField

from classificator import SentimentClassificator

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="School21",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))

model = SentimentClassificator()


class MyForm(FlaskForm):
    text = StringField('Enter the text', validators=[DataRequired()])


@app.route('/')
def hello_world():
    form = MyForm()
    if form.validate_on_submit():
        return "Form submitted"
    return render_template('submit.html', form=form)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    try:
        result = model.predict_sentiment(request.form['text'])
        if result == "Negative sentiment":
            image = "static/negative.png"
        else:
            image = "static/positive.png"

        return render_template("response.html", result=result, image=image)
    except KeyError:
        return redirect(url_for('bad_request'))


@app.route('/badrequest400')
def bad_request():
    return abort(400)
