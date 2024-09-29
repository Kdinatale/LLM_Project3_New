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

@app.route('/bigram')
def bigram():
    return render_template('bigram.html')
    
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


def sentence(start_word, max_length):
    current_word = start_word
    sentence = [current_word]

    #Loops a number of times decided by max_length
    for _ in range(max_length - 1):
        if current_word not in model_bigram_news.index:
            break  # Stop if the current word is not in the table

        # Get the probabilities of the next words
        next_word_probs = model_bigram_news.loc[current_word]

        # Filter out words with zero probability
        next_word_probs = next_word_probs[next_word_probs > 0]

        if next_word_probs.empty:
            break  # Stop if there are no valid next words

        # Chooses the next word randomly based on probability distribution
        next_word = np.random.choice(next_word_probs.index, p=next_word_probs.values)

        # Adds new word to end of sentence
        sentence.append(next_word)
        current_word = next_word  # Move to the next word

        # Set '.' as a sentence terminator
        if next_word == '.':
          # Breaks if a '.' is the next probable token
          break

    return render_template('bigram_result.html', sentence=' '.join(sentence))
    
@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    # data = request.get_json(force=True)

    # try:
    #     features = data['features']
    # except KeyError:
    #     return jsonify(error="The 'features' key is missing from the request payload.")
    data = request.form['start_word']

    return sentence(data, 20)

if __name__ == '__main__':
    app.run(debug=True)