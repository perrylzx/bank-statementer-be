import pandas as pd
import json
import os
import re
from services.model import model
from sentence_transformers import util

def clean_description_for_matching(description):
    """Remove numbers, dates, numerical patterns, and common stopwords from description for better similarity matching"""
    # Remove common numerical patterns: card numbers, dates, transaction IDs, etc.
    cleaned = re.sub(r'\d+', ' ', description)  # Remove all digits
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)  # Remove special characters except letters and spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)      # Replace multiple spaces with single space
    
    # Define stopwords - common words that don't help with business identification
    stopwords = {
        # Location indicators
        'singapore', 'sgp', 'si', 'usa', 'irl', 'swe', 'ww', 'mp',
        # Common transaction terms
        'transaction', 'ref', 'bill', 'payment', 'transfer', 'debit', 'credit',
        # Common abbreviations and codes
        'pte', 'ltd', 'inc', 'co', 'corp', 'llc',
        # Time indicators (already removed numbers, but these text versions)
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        # Common payment terms
        'usd', 'sgd', 'eur', 'gbp', 'cad', 'aud',
        # Common filler words
        'the', 'and', 'or', 'of', 'at', 'in', 'on', 'for', 'with', 'by'
    }
    
    # Remove stopwords
    words = cleaned.split()
    filtered_words = [word for word in words if word.lower() not in stopwords and len(word) > 1]
    
    return ' '.join(filtered_words).strip().lower()

def getTransactionsService(df):
    tags_path = "tags.json"
    tagged_examples = []
    if os.path.exists(tags_path):
        with open(tags_path, "r") as f:
            tagged_examples = json.load(f)

    # Use cleaned descriptions for embeddings and matching
    known_descriptions_original = [item["description"].lower() for item in tagged_examples]
    known_descriptions_cleaned = [clean_description_for_matching(desc) for desc in known_descriptions_original]
    known_categories = [item["category"] for item in tagged_examples]
    known_embeddings = model.encode(known_descriptions_cleaned, convert_to_tensor=True) if known_descriptions_cleaned else None

    transactions = []

    for index, row in df.iterrows():
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

        category = []

        if known_embeddings is not None:
            # Use cleaned description for similarity matching
            cleaned_description = clean_description_for_matching(description)
            tx_embedding = model.encode(cleaned_description, convert_to_tensor=True)
            cosine_scores = util.cos_sim(tx_embedding, known_embeddings)[0]
            best_match_idx = cosine_scores.argmax().item()
            best_score = cosine_scores[best_match_idx].item()

            if best_score > 0.7:
                category = known_categories[best_match_idx]

        transactions.append({
            "id": index,
            "date": str(date),
            "type": row["Reference"],
            "description": description,  # Keep original description for display
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

    # Avoid duplicate tags (check based on cleaned description)
    cleaned_ref_desc = clean_description_for_matching(ref_desc)
    already_exists = any(
        clean_description_for_matching(t["description"].lower()) == cleaned_ref_desc and t["category"] == ref_category
        for t in existing_tags
    )
    if not already_exists:
        existing_tags.append({
            "description": ref_desc,  # Store original description
            "category": ref_category
        })
        with open(tags_path, "w") as f:
            json.dump(existing_tags, f, indent=2)

    # Auto-categorize passed transactions using updated tag list
    known_descriptions_original = [t["description"].lower() for t in existing_tags]
    known_descriptions_cleaned = [clean_description_for_matching(desc) for desc in known_descriptions_original]
    known_categories = [t["category"] for t in existing_tags]
    known_embeddings = model.encode(known_descriptions_cleaned, convert_to_tensor=True)

    for tx in transactions:
        tx_desc = tx["description"].lower()
        # Use cleaned description for similarity matching
        cleaned_tx_desc = clean_description_for_matching(tx_desc)
        tx_embedding = model.encode(cleaned_tx_desc, convert_to_tensor=True)
        cosine_scores = util.cos_sim(tx_embedding, known_embeddings)[0]

        best_idx = cosine_scores.argmax().item()
        best_score = cosine_scores[best_idx].item()

        if best_score > 0.7:
            tx["category"] = known_categories[best_idx]

    return transactions
