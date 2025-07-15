from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained Random Forest model
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prepare the data in correct format
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the model
    prediction = model.predict(data)[0]

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
