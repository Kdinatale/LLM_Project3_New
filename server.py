from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the sentiment analysis pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load the bigram pre-trained model
model_bigram_news = pickle.load(open('bigram_news_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    features = request.form['sentence']

    prediction = model.polarity_scores(features)
    
    compound = float(prediction["compound"])
    
    return_string = "The Sentiment is "

    if compound > 0: 
        return_string += "Positive"
    else: 
        return_string += "Negative"
    
    return render_template('sentiment_result.html', result=str(compound), sentence=features)
    # return str(compound)
    

if __name__ == '__main__':
    app.run(debug=True)