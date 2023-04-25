
    
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning model and the CSV data
model = pickle.load(open("/Users/ruhimahendra/Desktop/Football_Transfermarkt/Machine_Learning_Engineering/model.pkl", "rb"))
data = pd.read_csv('/Users/ruhimahendra/Desktop/Football_Transfermarkt/Machine_Learning_Engineering/feed_model.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the ID from the form submission
    id = request.form['id']
    
    # Get the corresponding row from the data
    row = data[data['index'] == id]
    
    # Get the features from the row
    features = row.drop(['index'], axis=1)
    
    # Make a prediction using the machine learning model
    prediction = model.predict(features)
    
    # Return the prediction to the user
    return f'The predicted value for ID {id} is {prediction[0]}.'

if __name__ == '__main__':
    app.run(debug=True)
