import os, json, time
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from groq import Groq

HF_TOKEN        = os.environ["HF_TOKEN"]
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
HF_DATASET_REPO = "Akshu2424/rick-llm-dataset-2026"
TRANSCRIPT_CSV  = "./data/RickAndMortyScripts.csv"
OUTPUT_JSONL    = "./data/rick_dataset.jsonl"
ENRICH_N        = 200  # set to None to enrich all

SYSTEM = (
    "You are Rick Sanchez — sarcastic, nihilistic, brilliant. "
    "You burp mid-sentence (*burp*), reference science and the multiverse. "
    "Use 'Morty', 'wubba lubba dub dub'. Keep responses 1-4 sentences."
)

client = Groq(api_key=GROQ_API_KEY)


def build_pairs(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    cols = df.columns.tolist()
    char_col = next((c for c in cols if c in ["character", "name", "speaker"]), cols[0])
    line_col  = next((c for c in cols if c in ["line", "text", "dialogue"]), cols[1])

    pairs, buf = [], []
    for _, row in df.iterrows():
        char = str(row[char_col]).strip().upper()
        line = str(row[line_col]).strip()
        if not line or line == "nan":
            continue
        if "RICK" in char:
            if len(line) >= 25 and buf:
                human = " | ".join(buf)[-400:]
                pairs.append({"human": human, "rick": line})
            buf = []
        else:
            buf.append(f"{char}: {line}")

    print(f"Extracted {len(pairs)} pairs")
    return pairs


def enrich(pairs):
    sample = pairs[:ENRICH_N] if ENRICH_N else pairs
    rest   = pairs[ENRICH_N:] if ENRICH_N else []
    out = []
    for i, p in enumerate(sample):
        if i % 50 == 0:
            print(f"Enriching {i}/{len(sample)}...")
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": (
                        f"Rewrite this Rick line to be more in-character.\n"
                        f"Context: {p['human']}\nOriginal: {p['rick']}\nRewritten:"
                    )},
                ],
                max_tokens=150, temperature=0.8,
            )
            out.append({"human": p["human"], "rick": r.choices[0].message.content.strip()})
            time.sleep(0.1)
        except Exception as e:
            print(f"Groq error {i}: {e}")
            out.append(p)
    return out + rest


def main():
    pairs = build_pairs(TRANSCRIPT_CSV)
    pairs = enrich(pairs)

    samples = [
        {"conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human",  "value": p["human"]},
            {"from": "gpt",    "value": p["rick"]},
        ]} for p in pairs
    ]

    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} samples")

    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})
    print(f"Train: {len(dd['train'])} | Val: {len(dd['validation'])}")

    login(token=HF_TOKEN)
    dd.push_to_hub(HF_DATASET_REPO, private=False)
    print(f"Pushed to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
