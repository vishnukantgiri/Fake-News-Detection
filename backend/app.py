from flask import Flask, request, jsonify, send_from_directory
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data["news"]

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]
    confidence = model.predict_proba(news_vector)[0].max() * 100

    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run()