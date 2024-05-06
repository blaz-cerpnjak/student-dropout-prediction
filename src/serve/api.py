from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify("pong")

@app.route('/predict', methods=['POST'])
def predict():
    # Get json data
    json = request.get_json()
    if json is None:
        return jsonify({"error": "json data missing"}), 400
    
    # Convert json to dataframe
    df = pd.DataFrame(json)

    # Load model
    model = joblib.load('models/model.joblib')

    # Make prediction
    prediction = model.predict(df.head(1))    
    return jsonify({"prediction": prediction[0]}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)