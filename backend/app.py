from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# ==============================
# Load trained model
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "model.pkl"), "rb")
)

vectorizer = pickle.load(
    open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb")
)


# ==============================
# Frontend
# ==============================

@app.route("/")
def home():

    frontend_path = os.path.abspath(
        os.path.join(BASE_DIR, "../frontend")
    )

    return send_from_directory(
        frontend_path,
        "index.html"
    )


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

        news = str(data["news"]).strip()

        if not news:
            return jsonify({
                "error": "News text cannot be empty"
            }), 400

        # ==============================
        # Convert news into TF-IDF
        # ==============================

        news_vector = vectorizer.transform([news])

        # ==============================
        # Prediction probabilities
        # ==============================

        probabilities = model.predict_proba(news_vector)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        prediction = model.predict(news_vector)[0]

        # ==============================
        # Result
        # ==============================

        if prediction == 1:
            result = "REAL NEWS"
            confidence = real_probability
        else:
            result = "FAKE NEWS"
            confidence = fake_probability

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2),
            "fake_probability": round(fake_probability, 2),
            "real_probability": round(real_probability, 2)
        })

    except Exception as e:

        print("Prediction Error:", e)

        return jsonify({
            "error": "Something went wrong while detecting the news."
        }), 500


# ==============================
# Health Check
# ==============================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": "loaded"
    })


# ==============================
# Run Flask Server
# ==============================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )