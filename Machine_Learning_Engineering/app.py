
    
import Flask
import pandas as pd
import pickle

app = Flask(__name__)

# Load the machine learning model and the CSV data
model = pickle.load(open("/Users/ruhimahendra/Desktop/Football_Transfermarkt/Machine_Learning_Engineering/model.pkl", "rb"))
with open('/Users/ruhimahendra/Desktop/Football_Transfermarkt/Machine_Learning_Engineering/my_list.pkl', 'rb') as data:
    my_array = pickle.load(data)


@app.route('/')
def index():
    return Flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the ID from the form submission
    id = Flask.request.form['id']
    
    # Get the corresponding row from the data
    row = my_array[int(id)]
    
    # Get the features from the row
    features_array = row.reshape(1,-1)
    print(features_array.shape)
    
    # Make a prediction using the machine learning model
    prediction = model.predict(features_array)    
    # Return the prediction to the user
    return f'The predicted value for ID {id} is {prediction[0]}.'

if __name__ == '__main__':
    app.run(debug=True)
