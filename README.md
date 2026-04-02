# NLP and Text Analysis in Finance - MAEC Step 2 Feature Engineering

This workspace is configured for **step 2 (feature engineering)** using the MAEC earnings-call dataset.

## What we did

- Cloned the MAEC upstream repository into `MAEC_upstream/` and validated its folder structure.
- Implemented a full feature-engineering pipeline in `src/maec_feature_engineering.py`.
- Cleaned transcript text by removing noise markers (`<UNK>`, transcript artifacts) and normalizing whitespace.
- Built NLP linguistic variables including personalism index, certainty score, lexical/style metrics, and a sentiment proxy.
- Aggregated features at three levels: sentence, person, and call.
- Added a CEO-candidate identification module to flag likely management/CEO speaker per call.
- Merged call-level aggregates with low-level audio summary statistics from MAEC `features.csv` files.
- Produced processed datasets in `data/processed/`:
  - `sentences_cleaned_features.csv`
  - `person_level_features.csv`
  - `call_level_features.csv`
  - `ceo_candidates_by_call.csv`
- Ran an end-to-end sanity check and confirmed all output files were generated successfully.

## What this pipeline does

- Loads MAEC transcripts from `MAEC_Dataset`.
- Uses `MAEC_Dataset_Person_Label` speaker labels when available.
- Cleans sentence text (removes `<UNK>`, normalizes spacing, handles noisy transcript markers).
- Builds linguistic variables for NLP analysis:
  - `personalism_index` (singular/plural first person intensity)
  - `Loughran‑McDonald financial dictionary` (positive, negative, uncertainty, strong/weak modal, litigious, constraining)
  - `certainty_score` (based on LM certainty minus uncertainty, normalized by sentence length)
  - lexical and style metrics (`type_token_ratio`, `avg_token_len`, `hedge_count`, `negation_count`)
  - simple financial sentiment proxy (`sentiment_proxy`)
- Aggregates outputs at sentence, speaker (`person`), and call levels.
- Scores each speaker within a call and flags a single likely CEO/management candidate.
- Merges aggregated low-level audio statistics from `features.csv` into call-level data.

## Project structure

- `src/maec_feature_engineering.py`: End-to-end cleaning + linguistic variable construction.
- `data/processed/`: Output CSV files.
- `notebooks/ceo_candidate_sanity_table.md`: Quick sanity table for flagged speakers by ticker/date.
- `MAEC_upstream/`: Cloned source MAEC repository.

## Setup

```bash
pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_sm
```

## Run

```bash
python src/maec_feature_engineering.py
```

Optional explicit paths:

```bash
python src/maec_feature_engineering.py \
  --maec-dataset-dir MAEC_upstream/MAEC_Dataset \
  --person-label-dir MAEC_upstream/MAEC_Dataset_Person_Label \
  --output-dir data/processed
```

## Outputs

- `data/processed/sentences_cleaned_features.csv`
- `data/processed/person_level_features.csv`
- `data/processed/call_level_features.csv`
- `data/processed/ceo_candidates_by_call.csv`
- `data/processed/ceo_candidate_sanity_table.csv`

### CEO-candidate columns

- `person_level_features.csv` now includes:
  - `speaker_sentence_share`
  - `ceo_candidate_score`
  - `ceo_candidate_rank`
  - `is_ceo_candidate` (1 for the top speaker per call)
- `call_level_features.csv` now includes:
  - `ceo_candidate_person`
  - `ceo_candidate_score_call`
  - `ceo_candidate_sentence_share`
  - `ceo_candidate_n_sentences`
- `ceo_candidates_by_call.csv` is a compact call-level mapping for downstream modeling.

## Notes for your finance project

For your main hypothesis (CEO communication style vs short-term stock movement), start from:

1. `call_level_features.csv` for prediction/regression baselines.
2. `person_level_features.csv` to isolate management-style effects and identify likely lead speakers.
3. Merge with market/event variables (abnormal returns, surprises, controls) for final econometric models.

## Next plan

1. Add a modeling-ready merge template for abnormal returns + financial controls.
2. Add a baseline regression/classification notebook for your course deliverable.
