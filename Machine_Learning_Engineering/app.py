
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning model and the CSV data
model = pickle.load(open("model_new.pkl", "rb"))
with open("X_pca.pickle", 'rb') as data:
    my_array = pickle.load(data)

print(len(my_array))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the ID from the form submission
    id = request.form['id']
    
    # Get the corresponding row from the data
    row = my_array[int(id)]
    
    # Get the features from the row
    features_array = row.reshape(1,-1)
    print(features_array.shape)
    
    # Make a prediction using the machine learning model
    prediction = model.predict(features_array)    
    # Return the prediction to the user
    return f'The predicted value for ID {id} is €{round(prediction[0],2)}.'

if __name__ == '__main__':
    app.run(debug=True)
