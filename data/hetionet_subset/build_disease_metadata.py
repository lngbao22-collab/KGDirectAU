import pandas as pd
import wikipediaapi
import time
from pathlib import Path

# =====================================================
# Configuration
# =====================================================

INPUT_FILE = "hetionet-v1.0-nodes.tsv"
OUTPUT_FILE = "disease_metadata.tsv"

REQUEST_DELAY = 0.25

# =====================================================
# Wikipedia API
# =====================================================

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="KGAU-Hetionet-Metadata/1.0 (Academic Research)"
)

# =====================================================
# Manual aliases
# =====================================================

ALIASES = {
    "Parkinson disease": "Parkinson's disease",
    "Huntington disease": "Huntington's disease",
    "type 1 diabetes mellitus": "Type 1 diabetes",
    "type 2 diabetes mellitus": "Type 2 diabetes",
    "age related macular degeneration": "Age-related macular degeneration",
    "Lou Gehrig disease": "Amyotrophic lateral sclerosis",
}

# =====================================================
# Candidate title generation
# =====================================================

def candidate_titles(name):
    """
    Generate possible Wikipedia titles.
    """

    names = []

    name = ALIASES.get(name, name)

    names.append(name)

    # Remove parentheses
    if "(" in name:
        names.append(name.split("(")[0].strip())

    # Replace disease -> disease'
    if " disease" in name.lower() and "'" not in name:
        names.append(name.replace(" disease", "'s disease"))

    # Capitalize first letter
    names.append(name.title())

    # Remove duplicates
    seen = set()
    out = []

    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)

    return out

# =====================================================
# Retrieve metadata
# =====================================================

def retrieve(name):

    for title in candidate_titles(name):

        page = wiki.page(title)

        if page.exists():

            summary = page.summary.strip()

            if summary == "":
                summary = "Description unavailable."

            return {
                "wiki_title": page.title,
                "description": summary.replace("\n", " "),
                "wiki_url": page.fullurl,
            }

    return {
        "wiki_title": "",
        "description": "Description not found.",
        "wiki_url": "",
    }

# =====================================================
# Load data
# =====================================================

if Path(OUTPUT_FILE).exists():

    print("Loading cached metadata...")

    df = pd.read_csv(OUTPUT_FILE, sep="\t")

else:

    print("Loading diseases...")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    for col in ["wiki_title", "description", "wiki_url"]:
        if col not in df.columns:
            df[col] = ""

# =====================================================
# Main loop
# =====================================================

total = len(df)

print(f"Found {total} diseases.\n")

for idx, row in df.iterrows():

    if str(row["description"]).strip():
        continue

    disease = row["name"]

    print(f"[{idx+1}/{total}] {disease}")

    meta = retrieve(disease)

    df.at[idx, "wiki_title"] = meta["wiki_title"]
    df.at[idx, "description"] = meta["description"]
    df.at[idx, "wiki_url"] = meta["wiki_url"]

    # Save after every disease
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    time.sleep(REQUEST_DELAY)

print("\nDone!")
print(f"Saved to {OUTPUT_FILE}")