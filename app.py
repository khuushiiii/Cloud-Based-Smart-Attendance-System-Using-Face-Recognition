from flask import Flask, render_template, jsonify
from firebase_config import db

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/attendance")
def get_attendance():
    docs = db.collection("attendance").order_by("timestamp").stream()
    records = []
    for doc in docs:
        record = doc.to_dict()
        # Remove raw timestamp field (not JSON serializable)
        record.pop("timestamp", None)
        records.append(record)
    return jsonify(records)


@app.route("/api/today")
def get_today():
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    docs = db.collection("attendance").where("date", "==", today).stream()
    records = [doc.to_dict() for doc in docs]
    for r in records:
        r.pop("timestamp", None)
    return jsonify(records)


if __name__ == "__main__":
    app.run(debug=True)
