from flask import Blueprint, jsonify
import pandas as pd
from services.transaction_service import getTransactionsService

transactions_bp = Blueprint("transactions", __name__)

@transactions_bp.route("/transactions", methods=["GET"])
def getTransactions():
    df = pd.read_csv("your_statement.csv", on_bad_lines="skip", index_col=False)
    transactions = getTransactionsService(df)

    return jsonify({"transactions": transactions})
