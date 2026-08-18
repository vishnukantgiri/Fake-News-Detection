from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# ==============================
# Load trained model
# ==============================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# ==============================
# Frontend
# ==============================

@app.route("/")
def home():
    frontend_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../frontend")
    )

    return send_from_directory(frontend_path, "index.html")


# ==============================
# Fake News Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "news" not in data:
            return jsonify({
                "error": "News text is required"
            }), 400

        news = data["news"].strip()

        if not news:
            return jsonify({
                "error": "News text cannot be empty"
            }), 400

        # Convert news into vector
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)[0]

        # Confidence
        confidence = model.predict_proba(
            news_vector
        )[0].max() * 100

        # Result
        if prediction == 1:
            result = "REAL NEWS"
        else:
            result = "FAKE NEWS"

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("Prediction Error:", e)

        return jsonify({
            "error": "Something went wrong while detecting the news."
        }), 500


# ==============================
# Run Flask Server
# ==============================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )