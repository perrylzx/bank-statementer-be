from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from services.transaction_service import getTransactionsService, categorizeAndUpdate

app = Flask(__name__)

# Enable CORS for your Next.js frontend
CORS(app, origins=["https://bank-statementer-fe.vercel.app"])


@app.route("/transactions", methods=["POST"])
def getTransactions():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file, on_bad_lines="skip", index_col=False)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400
    transactions = getTransactionsService(df)

    return jsonify({"transactions": transactions})

@app.route("/categorize", methods=["POST"])
def categorize():
    data = request.get_json()
    reference = data.get("reference_transaction")
    transactions = data.get("transactions", [])

    if not reference or not transactions:
        return jsonify({"error": "Missing reference_transaction or transactions"}), 400

    ref_desc = reference.get("description", "").strip().lower()
    ref_category = reference.get("category", "").strip()
    if not ref_desc or not ref_category:
        return jsonify({"error": "Description and category required"}), 400

    transactions = categorizeAndUpdate(ref_desc=ref_desc, ref_category=ref_category, transactions=transactions)

    return jsonify({"transactions": transactions})


@app.route("/ping")
def ping():
    return "Latest code (deployed)", 200
