import json
import math
from collections import Counter
from typing import Literal, Dict

import pandas as pd
import requests


CSV_PATH = "reviews_most_relevant.csv"  
ENCODING = "latin1"  
TEXT_COLUMN = "en_full_review" 

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"  

AspectLabel = Literal["positive", "negative", "not_mentioned"]

SYSTEM_PROMPT = """
You are an aspect-based sentiment classifier for hotel reviews.

You will receive ONE hotel review, possibly in French or English.

You must classify THREE aspects:

1. cleanliness  (cleanliness of the room, bathroom, sheets, general hygiene)
2. comfort_equipment (bed, noise level, air conditioning, Wi-Fi, furniture, modern equipment)
3. location (location of the hotel, distance to center, transport, neighborhood)

For EACH aspect, choose ONE of:
- "positive"
- "negative"
- "not_mentioned"

Important:
- Consider indirect references. For example:
  - "la chambre était nickel" => cleanliness: positive
  - "linge douteux" => cleanliness: negative
  - "quartier calme, proche du centre" => location: positive
- If the aspect is not clearly mentioned, use "not_mentioned".
- If both positive and negative are present, choose the dominant overall feeling.

Your answer MUST be ONLY valid JSON, with this exact structure:

{
  "cleanliness": "positive | negative | not_mentioned",
  "comfort_equipment": "positive | negative | not_mentioned",
  "location": "positive | negative | not_mentioned"
}
"""


def classify_review_aspects_with_ollama(text: str) -> Dict[str, AspectLabel]:
  
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "format": "json",  # on demande du JSON directement
        "stream": False,
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    content = data["message"]["content"]
    parsed = json.loads(content)

   
    return {
        "cleanliness": parsed.get("cleanliness", "not_mentioned"),
        "comfort_equipment": parsed.get("comfort_equipment", "not_mentioned"),
        "location": parsed.get("location", "not_mentioned"),
    }


def compute_stats(labels: list[AspectLabel]):
    c = Counter(labels)
    total = len(labels)
    mentions = c["positive"] + c["negative"]

    if total == 0:
        return {
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "not_mentioned_pct": 0.0,
            "total_mentions": 0,
        }

    if mentions == 0:
        return {
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "not_mentioned_pct": 100.0,
            "total_mentions": 0,
        }

    positive_pct = 100 * c["positive"] / mentions
    negative_pct = 100 * c["negative"] / mentions
    not_mentioned_pct = 100 * c["not_mentioned"] / total

    return {
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "not_mentioned_pct": not_mentioned_pct,
        "total_mentions": mentions,
    }




def main():
    print("Chargement du CSV...")
    df = pd.read_csv(CSV_PATH, encoding=ENCODING)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"La colonne {TEXT_COLUMN} n'existe pas dans le CSV. Colonnes dispo : {list(df.columns)}")

    # nouvelles colonnes pour stocker les labels par review
    df["cleanliness_label"] = None
    df["comfort_label"] = None
    df["location_label"] = None

    cleanliness_labels = []
    comfort_labels = []
    location_labels = []

    print(f"Nombre de reviews : {len(df)}")

    for i, row in df.iterrows():
        text = str(row[TEXT_COLUMN])
        if not isinstance(text, str) or text.strip() == "" or text == "nan":
            df.at[i, "cleanliness_label"] = "not_mentioned"
            df.at[i, "comfort_label"] = "not_mentioned"
            df.at[i, "location_label"] = "not_mentioned"
            cleanliness_labels.append("not_mentioned")
            comfort_labels.append("not_mentioned")
            location_labels.append("not_mentioned")
            continue

        print(f"\n--- Review {i+1}/{len(df)} ---")
        print(text[:200].replace("\n", " ") + ("..." if len(text) > 200 else ""))

        try:
            result = classify_review_aspects_with_ollama(text)
        except Exception as e:
            print("Erreur avec Ollama, on marque tout en not_mentioned :", e)
            result = {
                "cleanliness": "not_mentioned",
                "comfort_equipment": "not_mentioned",
                "location": "not_mentioned",
            }

        cl = result["cleanliness"]
        co = result["comfort_equipment"]
        lo = result["location"]

        df.at[i, "cleanliness_label"] = cl
        df.at[i, "comfort_label"] = co
        df.at[i, "location_label"] = lo

        cleanliness_labels.append(cl)
        comfort_labels.append(co)
        location_labels.append(lo)

        print(" → cleanliness:", cl, "| comfort:", co, "| location:", lo)

    stats_cleanliness = compute_stats(cleanliness_labels)
    stats_confort = compute_stats(comfort_labels)
    stats_location = compute_stats(location_labels)

    print("\n================= RÉSULTATS GLOBAUX =================")
    print("\nPropreté :")
    print(stats_cleanliness)

    print("\nConfort / équipements :")
    print(stats_confort)

    print("\nEmplacement :")
    print(stats_location)


    out_path = "reviews_with_aspects.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nFichier sauvegardé avec labels : {out_path}")


if __name__ == "__main__":
    main()
