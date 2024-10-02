from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(message)s', filename='server_log.log', encoding='utf-8')
logger.setLevel(logging.DEBUG)
logger.debug("TESTING")

app = Flask(__name__)

# Load the sentiment analysis pre-trained model
try:
    file1 = 'model.pkl'
    model = pickle.load(open(file1, 'rb'))
except FileNotFoundError:
    logger.error(f"The file {file1} was not found")
except pickle.PicklingError as e:
    logger.error(f"A Pickling Error {e} Occured: An unpickleable object is encountered by the pickler.")
except pickle.UnpicklingError as e:
    logger.error(f"A Unpickling Error {e} Occured: An error encountered when unpickling an object.")
except Exception as e:
    # Included because the python documentation says "other exceptions may also be raised during unpickling, including (but not necessarily limited to) AttributeError, EOFError, ImportError, and IndexError."
    logger.error(f"The error is {e}")
else:
    logger.info(f"The file {file1} was loaded without any errors")
    

try:
    # Load the bigram pre-trained model
    file2 = 'bigram_news_model.pkl'
    model_bigram_news = pickle.load(open(file2, 'rb'))
except FileNotFoundError:
    logger.error(f"The file {file2} was not found")
except pickle.PicklingError as e:
    logger.error(f"A Pickling Error {e} Occured: An unpickleable object is encountered by the pickler.")
except pickle.UnpicklingError as e:
    logger.error(f"A Unpickling Error {e} Occured: An error encountered when unpickling an object.")
except Exception as e:
    # Included because the python documentation says "other exceptions may also be raised during unpickling, including (but not necessarily limited to) AttributeError, EOFError, ImportError, and IndexError."
    logger.error(f"The error is {e}")
else:
    logger.info(f"The file {file2} was loaded without any errors")



@app.route('/')
def home():
    logger.info("The route / is called")
    return render_template('index.html')

@app.route('/sentiment')
def sentiment():
    logger.info("The route /sentiment is called")
    return render_template('sentiment.html')

@app.route('/bigram')
def bigram():
    logger.info("The route /bigram is called")
    return render_template('bigram.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    logger.info("The route /predict is called")
    features = request.form['sentence']
    logger.info(f"The data '{features}' is recieved from the POST request")

    prediction = model.polarity_scores(features)
    
    compound = float(prediction["compound"])
    
    return_string = "The Sentiment is "

    if compound > 0: 
        return_string += "Positive"
    else: 
        return_string += "Negative"
    
    return render_template('sentiment_result.html', result=str(compound), sentence=features)
    # return str(compound)


def sentence(start_word):
    logger.info("The helper function sentence() is called")
    logger.info(f"The start_word '{start_word}' was passed as a parameter")
    current_word = start_word.lower()
    current_word_capital = current_word[0].upper() + current_word[1:]
    sentence = [current_word_capital]

    #Loops a number of times decided by max_length
    loop = True
    while loop:
    # for _ in range(max_length - 1):
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

        # Set '.' as a sentence terminator
        if next_word == '.':
          # Breaks if a '.' is the next probable token
          loop = False
          break
        
        sentence.append(next_word)
        current_word = next_word  # Move to the next word

    return render_template('bigram_result.html', sentence=' '.join(sentence) + '.', start_word = current_word_capital)
    
@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    # data = request.get_json(force=True)

    # try:
    #     features = data['features']
    # except KeyError:
    #     return jsonify(error="The 'features' key is missing from the request payload.")
    logger.info("The route /generate_sentence is called")
    data = request.form['start_word']
    logger.info(f"The data '{data}' is recieved from the POST request")

    return sentence(data)

@app.errorhandler(404)
def not_found(e):
    app.logger.error(f'Page not found: {request.path}')
    return render_template('not_found.html', path = request.path), 404

if __name__ == '__main__':
    app.run(debug=True)