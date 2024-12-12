from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import Normalizer

app = Flask(__name__)

# Load the trained model
with open("random_forest_diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Normalizer for consistent preprocessing
columns_to_normalize = ['BMI', 'MentHlth', 'GenHlth', 'PhysHlth', 'Age', 'Education', 'Income']
normalizer = Normalizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse data from the form
    input_data = request.form.to_dict()
    input_data = {k: float(v) for k, v in input_data.items()}

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply normalization
    input_df[columns_to_normalize] = normalizer.fit_transform(input_df[columns_to_normalize])

    # Make prediction
    prediction = model.predict(input_df)[0]
    result_map = {0: "No Diabetes", 1: "Diabetes", 2: "Prediabetes"}
    result = result_map.get(prediction, "Unknown Result")

    # Structure the result message more clearly
    return jsonify({
        "Prediction": result,
        "Message": f"The model predicts: {result}. Please follow up with a healthcare provider for a thorough diagnosis."
    })


if __name__ == '__main__':
    app.run(debug=True)
