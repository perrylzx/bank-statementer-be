import pandas as pd
import json
import os
from services.model import model
from sentence_transformers import util

def getTransactionsService(df):
    tags_path = "tags.json"
    tagged_examples = []
    if os.path.exists(tags_path):
        with open(tags_path, "r") as f:
            tagged_examples = json.load(f)

    known_descriptions = [item["description"].lower() for item in tagged_examples]
    known_categories = [item["category"] for item in tagged_examples]
    known_embeddings = model.encode(known_descriptions, convert_to_tensor=True) if known_descriptions else None

    transactions = []

    for _, row in df.iterrows():
        debit_str = str(row["Debit Amount"]).replace(",", "").strip()
        credit_str = str(row["Credit Amount"]).replace(",", "").strip()

        if not debit_str and not credit_str:
            continue

        try:
            amount = -float(debit_str) if debit_str else float(credit_str)
        except ValueError:
            continue

        try:
            date = pd.to_datetime(row["Transaction Date"], dayfirst=False).date()
        except Exception:
            continue

        description = " ".join(
            str(x).strip()
            for x in [row.get("Transaction Ref1"), row.get("Transaction Ref2"), row.get("Transaction Ref3")]
            if pd.notna(x)
        ).lower()

        category = "Uncategorized"

        if known_embeddings is not None:
            tx_embedding = model.encode(description, convert_to_tensor=True)
            cosine_scores = util.cos_sim(tx_embedding, known_embeddings)[0]
            best_match_idx = cosine_scores.argmax().item()
            best_score = cosine_scores[best_match_idx].item()

            if best_score > 0.7:
                category = known_categories[best_match_idx]

        transactions.append({
            "date": str(date),
            "type": row["Reference"],
            "description": description,
            "amount": amount,
            "category": category
        })

    return transactions

def categorizeAndUpdate(ref_desc, ref_category, transactions):
    tags_path = "tags.json"
    existing_tags = []
    if os.path.exists(tags_path):
        with open(tags_path, "r") as f:
            try:
                existing_tags = json.load(f)
            except Exception:
                existing_tags = []

    # Avoid duplicate tags
    already_exists = any(
        t["description"].lower() == ref_desc and t["category"] == ref_category
        for t in existing_tags
    )
    if not already_exists:
        existing_tags.append({
            "description": ref_desc,
            "category": ref_category
        })
        with open(tags_path, "w") as f:
            json.dump(existing_tags, f, indent=2)

    # Auto-categorize passed transactions using updated tag list
    known_descriptions = [t["description"].lower() for t in existing_tags]
    known_categories = [t["category"] for t in existing_tags]
    known_embeddings = model.encode(known_descriptions, convert_to_tensor=True)

    for tx in transactions:
        tx_desc = tx["description"].lower()
        tx_embedding = model.encode(tx_desc, convert_to_tensor=True)
        cosine_scores = util.cos_sim(tx_embedding, known_embeddings)[0]

        best_idx = cosine_scores.argmax().item()
        best_score = cosine_scores[best_idx].item()

        if best_score > 0.7:
            tx["category"] = known_categories[best_idx]

    return transactions
