from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Advanced sentiment analysis
from textblob import TextBlob

# For improved personalism index using spaCy
import spacy

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    raise

# ================================
# Loughran-McDonald Financial Dictionary
# ================================
LM_DICT_PATH = "Loughran-McDonald_MasterDictionary_1993-2024.csv"   # Update path if necessary

LM_POSITIVE = set()
LM_NEGATIVE = set()
LM_UNCERTAINTY = set()
LM_STRONG_MODAL = set()
LM_WEAK_MODAL = set()
LM_LITIGIOUS = set()
LM_CONSTRAINING = set()

try:
    # Read CSV, force 'Word' column as string
    lm_df = pd.read_csv(LM_DICT_PATH, dtype={'Word': str}, encoding='utf-8')
    
    required_cols = ['Word', 'Negative', 'Positive', 'Uncertainty',
                     'Strong_Modal', 'Weak_Modal', 'Litigious', 'Constraining']
    missing = [col for col in required_cols if col not in lm_df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}. Using default empty sets.")
    else:
        # Convert sentiment columns to numeric (non-numeric -> NaN), then fill NaN with 0
        for col in required_cols[1:]:   # skip 'Word'
            lm_df[col] = pd.to_numeric(lm_df[col], errors='coerce').fillna(0)

        for _, row in lm_df.iterrows():
            word = row['Word']
            # Skip invalid words
            if pd.isna(word) or not isinstance(word, str):
                continue
            word = word.lower().strip()
            if not word:
                continue

            # Add to sets if value is non-zero (2009 or any non-zero)
            if row['Positive'] != 0:
                LM_POSITIVE.add(word)
            if row['Negative'] != 0:
                LM_NEGATIVE.add(word)
            if row['Uncertainty'] != 0:
                LM_UNCERTAINTY.add(word)
            if row['Strong_Modal'] != 0:
                LM_STRONG_MODAL.add(word)
            if row['Weak_Modal'] != 0:
                LM_WEAK_MODAL.add(word)
            if row['Litigious'] != 0:
                LM_LITIGIOUS.add(word)
            if row['Constraining'] != 0:
                LM_CONSTRAINING.add(word)

    print(f"Loaded LM dictionary: Positive={len(LM_POSITIVE)}, Negative={len(LM_NEGATIVE)}, "
          f"Uncertainty={len(LM_UNCERTAINTY)}, Strong_Modal={len(LM_STRONG_MODAL)}, "
          f"Weak_Modal={len(LM_WEAK_MODAL)}, Litigious={len(LM_LITIGIOUS)}, Constraining={len(LM_CONSTRAINING)}")

except FileNotFoundError:
    print(f"Warning: {LM_DICT_PATH} not found. Loughran-McDonald features will be zero.")
except Exception as e:
    print(f"Error loading LM dictionary: {e}")
    print("Continuing with empty sets.")

# ================================
# Other dictionaries (Hedge, Negation)
# ================================
HEDGE_WORDS = {
    "sort",
    "kind",
    "roughly",
    "fairly",
    "relatively",
    "somewhat",
    "apparently",
    "likely",
    "unlikely",
    "perhaps",
}

NEGATION_WORDS = {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "can't"}

# ================================
# Text cleaning and tokenization
# ================================
TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")
MULTISPACE_PATTERN = re.compile(r"\s+")
UNK_PATTERN = re.compile(r"<UNK>", flags=re.IGNORECASE)
PAREN_SPEAKER_PATTERN = re.compile(r"\((multiple speakers|inaudible|ph|crosstalk)\)", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\n", " ")
    text = UNK_PATTERN.sub(" ", text)
    text = PAREN_SPEAKER_PATTERN.sub(" ", text)
    text = text.replace("--", " ")
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def count_in_set(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(1 for t in tokens if t in lexicon)


# ================================
# Sentence‑level feature extraction
# ================================
def sentence_features(sentence: str) -> Dict[str, float]:
    cleaned = clean_text(sentence)
    tokens = tokenize(cleaned)
    n_tokens = len(tokens)
    n_unique = len(set(tokens))

    # ---------- 1. Improved personalism index (spaCy) ----------
    doc = nlp(cleaned)
    first_person_singular = 0
    first_person_plural = 0
    for token in doc:
        if token.pos_ == "PRON":
            morph = token.morph
            if morph.get("Person") == ["1"]:
                if morph.get("Number") == ["Sing"]:
                    first_person_singular += 1
                elif morph.get("Number") == ["Plur"]:
                    first_person_plural += 1
                elif not morph.get("Number"):
                    first_person_singular += 1

    personalism_singular_ratio = first_person_singular / max(1, n_tokens)
    personalism_plural_ratio = first_person_plural / max(1, n_tokens)
    personalism_total_ratio = (first_person_singular + first_person_plural) / max(1, n_tokens)

    # ---------- 2. Loughran-McDonald features ----------
    lm_pos_count = count_in_set(tokens, LM_POSITIVE)
    lm_neg_count = count_in_set(tokens, LM_NEGATIVE)
    lm_uncertainty_count = count_in_set(tokens, LM_UNCERTAINTY)
    lm_strong_count = count_in_set(tokens, LM_STRONG_MODAL)
    lm_weak_count = count_in_set(tokens, LM_WEAK_MODAL)
    lm_litigious_count = count_in_set(tokens, LM_LITIGIOUS)
    lm_constraining_count = count_in_set(tokens, LM_CONSTRAINING)

    # Use LM Strong_Modal as "certainty" and LM Uncertainty as "uncertainty"
    certainty_count = lm_strong_count
    uncertainty_count = lm_uncertainty_count
    certainty_score = (certainty_count - uncertainty_count) / max(1, n_tokens)

    # Additional features
    hedge_count = count_in_set(tokens, HEDGE_WORDS)
    negation_count = count_in_set(tokens, NEGATION_WORDS)
    numeric_count = sum(1 for t in cleaned.split() if any(ch.isdigit() for ch in t))

    # Extreme positivity proxy: strong modals ratio
    extreme_positive_ratio = lm_strong_count / max(1, n_tokens)
    extreme_positive_ratio_adj = lm_strong_count / max(1, lm_pos_count) if lm_pos_count > 0 else 0.0

    # ---------- 3. TextBlob sentiment ----------
    blob = TextBlob(cleaned)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # ---------- 4. Return all features ----------
    return {
        "clean_sentence": cleaned,
        "n_chars": float(len(cleaned)),
        "n_tokens": float(n_tokens),
        "n_unique_tokens": float(n_unique),
        "type_token_ratio": float(n_unique / max(1, n_tokens)),
        "avg_token_len": float(np.mean([len(t) for t in tokens]) if tokens else 0.0),

        # Personalism
        "personal_pronoun_singular": float(first_person_singular),
        "personal_pronoun_plural": float(first_person_plural),
        "personalism_singular_ratio": float(personalism_singular_ratio),
        "personalism_plural_ratio": float(personalism_plural_ratio),
        "personalism_total_ratio": float(personalism_total_ratio),

        # Certainty/Uncertainty (now from LM)
        "certainty_count": float(certainty_count),
        "uncertainty_count": float(uncertainty_count),
        "certainty_score": float(certainty_score),
        "hedge_count": float(hedge_count),
        "negation_count": float(negation_count),

        # All Loughran-McDonald categories
        "lm_positive_count": float(lm_pos_count),
        "lm_negative_count": float(lm_neg_count),
        "lm_uncertainty_count": float(lm_uncertainty_count),
        "lm_strong_modal_count": float(lm_strong_count),
        "lm_weak_modal_count": float(lm_weak_count),
        "lm_litigious_count": float(lm_litigious_count),
        "lm_constraining_count": float(lm_constraining_count),
        "extreme_positive_ratio": extreme_positive_ratio,
        "extreme_positive_ratio_adj": extreme_positive_ratio_adj,

        # Numeric and punctuation
        "numeric_token_count": float(numeric_count),
        "has_question": float("?" in cleaned),
        "has_exclamation": float("!" in cleaned),

        # TextBlob sentiment
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
    }


# ================================
# Data processing, aggregation, and main function
# ================================
def parse_call_id(folder_name: str) -> Dict[str, str]:
    date_part = folder_name[:8]
    ticker = folder_name[9:] if "_" in folder_name else "UNKNOWN"
    return {"call_date": date_part, "ticker": ticker}


def load_person_labeled_sentences(person_file: Path) -> pd.DataFrame:
    df = pd.read_csv(person_file)
    expected = {"Person", "Sentence"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Unexpected columns in {person_file}: {list(df.columns)}")
    df = df.rename(columns={"Person": "person", "Sentence": "sentence"})
    return df[["person", "sentence"]]


def load_text_lines(text_file: Path) -> pd.DataFrame:
    lines = text_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = [{"person": "Unknown", "sentence": ln} for ln in lines if ln.strip()]
    return pd.DataFrame(rows)


def aggregate_audio_features(features_file: Path, call_id: str) -> pd.DataFrame:
    audio_df = pd.read_csv(features_file)
    audio_df = audio_df.replace("--undefined--", np.nan)
    audio_df.columns = [c.strip().lower().replace(" ", "_") for c in audio_df.columns]

    for col in audio_df.columns:
        audio_df[col] = pd.to_numeric(audio_df[col], errors="coerce")

    agg: Dict[str, float] = {"call_id": call_id}
    for col in audio_df.columns:
        if audio_df[col].notna().sum() == 0:
            continue
        agg[f"audio_mean_{col}"] = float(audio_df[col].mean())
        agg[f"audio_std_{col}"] = float(audio_df[col].std(ddof=0))

    return pd.DataFrame([agg])


def build_dataset(maec_dataset_dir: Path, person_label_dir: Path) -> pd.DataFrame:
    call_dirs = sorted([p for p in maec_dataset_dir.iterdir() if p.is_dir()])
    all_rows: List[pd.DataFrame] = []

    for call_dir in tqdm(call_dirs, desc="Processing MAEC calls"):
        call_id = call_dir.name
        text_file = call_dir / "text.txt"
        if not text_file.exists():
            continue

        person_file = person_label_dir / call_id / "text.csv"
        if person_file.exists():
            df = load_person_labeled_sentences(person_file)
        else:
            df = load_text_lines(text_file)

        if df.empty:
            continue

        meta = parse_call_id(call_id)
        df.insert(0, "call_id", call_id)
        df.insert(1, "call_date", meta["call_date"])
        df.insert(2, "ticker", meta["ticker"])
        df["sentence_index"] = np.arange(len(df), dtype=int)

        feat_df = df["sentence"].apply(sentence_features).apply(pd.Series)
        df = pd.concat([df, feat_df], axis=1)
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No valid MAEC transcript files were found.")

    return pd.concat(all_rows, ignore_index=True)


def aggregate_person_level(sent_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = sent_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "sentence_index"]

    grouped = (
        sent_df.groupby(["call_id", "call_date", "ticker", "person"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    person_counts = (
        sent_df.groupby(["call_id", "person"], dropna=False)
        .size()
        .reset_index(name="n_sentences_person")
    )
    return grouped.merge(person_counts, on=["call_id", "person"], how="left")


def aggregate_call_level(sent_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = sent_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "sentence_index"]

    call_level = sent_df.groupby(["call_id", "call_date", "ticker"], dropna=False)[numeric_cols].mean().reset_index()
    call_counts = sent_df.groupby("call_id", dropna=False).size().reset_index(name="n_sentences_call")
    speaker_counts = sent_df.groupby("call_id", dropna=False)["person"].nunique().reset_index(name="n_speakers_call")
    call_level = call_level.merge(call_counts, on="call_id", how="left")
    return call_level.merge(speaker_counts, on="call_id", how="left")


def _min_max_scale(series: pd.Series) -> pd.Series:
    s = series.fillna(0.0).astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v - min_v < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - min_v) / (max_v - min_v)


def identify_ceo_candidates(person_df: pd.DataFrame) -> pd.DataFrame:
    """Flag one likely management/CEO speaker per call using a transparent heuristic score."""
    df = person_df.copy()

    if "n_sentences_person" not in df.columns:
        raise ValueError("person_df must contain 'n_sentences_person' for CEO candidate identification")

    totals = df.groupby("call_id", dropna=False)["n_sentences_person"].transform("sum")
    df["speaker_sentence_share"] = df["n_sentences_person"] / totals.replace(0, np.nan)
    df["speaker_sentence_share"] = df["speaker_sentence_share"].fillna(0.0)

    certainty_base = (
        df["certainty_score"]
        if "certainty_score" in df.columns
        else pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    )
    personalism_base = (
        df["personalism_total_ratio"]
        if "personalism_total_ratio" in df.columns
        else pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    )
    question_base = (
        df["has_question"]
        if "has_question" in df.columns
        else pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    )

    certainty_component = certainty_base.groupby(df["call_id"], dropna=False).transform(_min_max_scale)
    personalism_component = personalism_base.groupby(df["call_id"], dropna=False).transform(_min_max_scale)
    question_component = question_base.groupby(df["call_id"], dropna=False).transform(_min_max_scale)

    # Heuristic weighting: dominant speakers with confident, first-person style and fewer question marks are preferred.
    df["ceo_candidate_score"] = (
        0.55 * df["speaker_sentence_share"]
        + 0.20 * certainty_component
        + 0.20 * personalism_component
        + 0.05 * (1.0 - question_component)
    )

    df["ceo_candidate_rank"] = (
        df.groupby("call_id", dropna=False)["ceo_candidate_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    df["is_ceo_candidate"] = (df["ceo_candidate_rank"] == 1).astype(int)

    return df


def build_ceo_candidate_map(person_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "call_id",
        "call_date",
        "ticker",
        "person",
        "ceo_candidate_score",
        "speaker_sentence_share",
        "n_sentences_person",
    ]
    ceo_map = person_df.loc[person_df["is_ceo_candidate"] == 1, cols].copy()
    ceo_map = ceo_map.rename(
        columns={
            "person": "ceo_candidate_person",
            "ceo_candidate_score": "ceo_candidate_score_call",
            "speaker_sentence_share": "ceo_candidate_sentence_share",
            "n_sentences_person": "ceo_candidate_n_sentences",
        }
    )
    return ceo_map


def merge_audio(call_level_df: pd.DataFrame, maec_dataset_dir: Path) -> pd.DataFrame:
    audio_rows: List[pd.DataFrame] = []
    for call_id in call_level_df["call_id"].tolist():
        features_file = maec_dataset_dir / call_id / "features.csv"
        if features_file.exists():
            audio_rows.append(aggregate_audio_features(features_file, call_id))

    if not audio_rows:
        return call_level_df

    audio_df = pd.concat(audio_rows, ignore_index=True)
    return call_level_df.merge(audio_df, on="call_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean MAEC transcripts and build linguistic variables for NLP feature engineering."
    )
    parser.add_argument(
        "--maec-dataset-dir",
        type=Path,
        default=Path("MAEC_upstream/MAEC_Dataset"),
        help="Path to MAEC_Dataset folder.",
    )
    parser.add_argument(
        "--person-label-dir",
        type=Path,
        default=Path("MAEC_upstream/MAEC_Dataset_Person_Label"),
        help="Path to MAEC_Dataset_Person_Label folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for engineered datasets.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sent_df = build_dataset(args.maec_dataset_dir, args.person_label_dir)
    person_df = aggregate_person_level(sent_df)
    person_df = identify_ceo_candidates(person_df)
    ceo_map_df = build_ceo_candidate_map(person_df)
    call_df = aggregate_call_level(sent_df)
    call_df = call_df.merge(ceo_map_df, on=["call_id", "call_date", "ticker"], how="left")
    call_df = merge_audio(call_df, args.maec_dataset_dir)

    sent_path = args.output_dir / "sentences_cleaned_features.csv"
    person_path = args.output_dir / "person_level_features.csv"
    call_path = args.output_dir / "call_level_features.csv"
    ceo_map_path = args.output_dir / "ceo_candidates_by_call.csv"

    sent_df.to_csv(sent_path, index=False)
    person_df.to_csv(person_path, index=False)
    call_df.to_csv(call_path, index=False)
    ceo_map_df.to_csv(ceo_map_path, index=False)

    print(f"Saved sentence-level data to {sent_path}")
    print(f"Saved person-level data to {person_path}")
    print(f"Saved call-level data to {call_path}")
    print(f"Saved CEO-candidate map to {ceo_map_path}")


if __name__ == "__main__":
    main()
