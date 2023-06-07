from flask import Flask, session, redirect, render_template, request
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, FileField
from wtforms.validators import DataRequired
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'

classifier = pickle.load(open('classifier.pkl', 'rb'))

class BankNote(FlaskForm):
    variance = FloatField('Enter variance', validators=[DataRequired()])
    skewness = FloatField('Enter skewness', validators=[DataRequired()])
    curtosis = FloatField('Enter curtosis', validators=[DataRequired()])
    entropy = FloatField('Enter entropy', validators=[DataRequired()])
    submit = SubmitField('Predict Bank Note')





@app.route('/', methods=['POST', "GET"])
def index():
    form = BankNote()

    if form.validate_on_submit():
        session['variance'] = form.variance.data
        session['skewness'] = form.skewness.data
        session['curtosis'] = form.curtosis.data
        session['entropy'] = form.entropy.data
        return redirect('result_single')
    
    return render_template('home.html', form = form)

@app.route('/upload_file', methods = ['GET', 'POST'])
def upload_file():
    return render_template('upload_file.html')


@app.route('/result_single', methods = ['GET', 'POST'])
def result_single():
    pred = classifier.predict([[session['variance'], session['skewness'], session['curtosis'], session['entropy']]])
    res = ''
    if pred[0] > 0.5:
        res = 'fake'
    else:
        res = 'real'

    return  render_template('result_single.html', pred = pred[0], res = res)




@app.route('/result_file', methods = ['GET', 'POST'])
def result_file():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        pred = classifier.predict(df)
        pred = list(pred)
        return render_template('result_file.html', pred = pred)


if __name__ == '__main__':
    app.run()





