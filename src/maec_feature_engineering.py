from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm


PERSONAL_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

CERTAINTY_WORDS = {
    "definitely",
    "certainly",
    "clearly",
    "always",
    "must",
    "will",
    "confident",
    "strong",
    "solid",
    "prove",
    "proven",
    "guarantee",
    "assured",
    "expect",
    "expected",
}

UNCERTAINTY_WORDS = {
    "maybe",
    "might",
    "could",
    "may",
    "uncertain",
    "uncertainty",
    "risk",
    "risks",
    "volatile",
    "volatility",
    "approximately",
    "around",
    "estimate",
    "estimates",
    "potential",
    "possible",
    "possibly",
    "challenging",
}

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

POSITIVE_WORDS = {
    "growth",
    "improve",
    "improved",
    "improvement",
    "strong",
    "benefit",
    "benefits",
    "efficient",
    "opportunity",
    "opportunities",
    "momentum",
    "outperform",
    "beat",
    "resilient",
}

NEGATIVE_WORDS = {
    "decline",
    "declined",
    "weak",
    "headwind",
    "headwinds",
    "loss",
    "losses",
    "risk",
    "risks",
    "pressure",
    "uncertain",
    "uncertainty",
    "deteriorated",
    "drag",
    "challenging",
}

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


def sentence_features(sentence: str) -> Dict[str, float]:
    cleaned = clean_text(sentence)
    tokens = tokenize(cleaned)
    n_tokens = len(tokens)
    n_unique = len(set(tokens))

    personal_count = count_in_set(tokens, PERSONAL_PRONOUNS)
    certainty_count = count_in_set(tokens, CERTAINTY_WORDS)
    uncertainty_count = count_in_set(tokens, UNCERTAINTY_WORDS)
    hedge_count = count_in_set(tokens, HEDGE_WORDS)
    negation_count = count_in_set(tokens, NEGATION_WORDS)
    positive_count = count_in_set(tokens, POSITIVE_WORDS)
    negative_count = count_in_set(tokens, NEGATIVE_WORDS)
    numeric_count = sum(1 for t in cleaned.split() if any(ch.isdigit() for ch in t))

    certainty_score = (certainty_count - uncertainty_count) / max(1, n_tokens)
    personalism_index = personal_count / max(1, n_tokens)
    sentiment_proxy = (positive_count - negative_count) / max(1, n_tokens)

    return {
        "clean_sentence": cleaned,
        "n_chars": float(len(cleaned)),
        "n_tokens": float(n_tokens),
        "n_unique_tokens": float(n_unique),
        "type_token_ratio": float(n_unique / max(1, n_tokens)),
        "avg_token_len": float(np.mean([len(t) for t in tokens]) if tokens else 0.0),
        "personal_pronoun_count": float(personal_count),
        "personalism_index": float(personalism_index),
        "certainty_count": float(certainty_count),
        "uncertainty_count": float(uncertainty_count),
        "certainty_score": float(certainty_score),
        "hedge_count": float(hedge_count),
        "negation_count": float(negation_count),
        "positive_word_count": float(positive_count),
        "negative_word_count": float(negative_count),
        "sentiment_proxy": float(sentiment_proxy),
        "numeric_token_count": float(numeric_count),
        "has_question": float("?" in cleaned),
        "has_exclamation": float("!" in cleaned),
    }


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
    call_df = aggregate_call_level(sent_df)
    call_df = merge_audio(call_df, args.maec_dataset_dir)

    sent_path = args.output_dir / "sentences_cleaned_features.csv"
    person_path = args.output_dir / "person_level_features.csv"
    call_path = args.output_dir / "call_level_features.csv"

    sent_df.to_csv(sent_path, index=False)
    person_df.to_csv(person_path, index=False)
    call_df.to_csv(call_path, index=False)

    print(f"Saved sentence-level data to {sent_path}")
    print(f"Saved person-level data to {person_path}")
    print(f"Saved call-level data to {call_path}")


if __name__ == "__main__":
    main()
