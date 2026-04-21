#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 MAEC 文件夹结构，使用 RoBERTa 模型计算每个 earnings call 的 Uncertainty 分数。
硬编码路径为 MAEC_upstream/MAEC_Dataset 和 MAEC_upstream/MAEC_Dataset_Person_Label。
输出 CSV: uncertainty_scores.csv (包含 call_id 和 uncertainty_roberta)
"""

import re
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ================================
# 硬编码路径（与原代码默认一致）
# ================================
MAEC_DATASET_DIR = Path("MAEC_upstream/MAEC_Dataset")
PERSON_LABEL_DIR = Path("MAEC_upstream/MAEC_Dataset_Person_Label")
OUTPUT_CSV = Path("uncertainty_scores.csv")

# ================================
# 文本清洗（与原代码完全一致）
# ================================
MULTISPACE_PATTERN = re.compile(r"\s+")
UNK_PATTERN = re.compile(r"<UNK>", flags=re.IGNORECASE)
PAREN_SPEAKER_PATTERN = re.compile(r"\((multiple speakers|inaudible|ph|crosstalk)\)", flags=re.IGNORECASE)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.replace("\n", " ")
    text = UNK_PATTERN.sub(" ", text)
    text = PAREN_SPEAKER_PATTERN.sub(" ", text)
    text = text.replace("--", " ")
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()

# ================================
# RoBERTa 模型加载
# ================================
MODEL_NAME = "NLPScholars/Roberta-Earning-Call-Transcript-Classification"
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 
print(f"使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

def get_uncertainty_score(text: str) -> float:
    """单条文本的 Uncertainty 概率（模型输出索引4）"""
    if not text or len(text.strip()) < 10:
        return np.nan
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=240,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)
    return probs[0][4].item()

def get_uncertainty_by_sentences(text: str) -> float:
    """按句子平均，避免长文本截断"""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return np.nan
    cleaned = clean_text(text)
    sentences = re.split(r'[.!?]+', cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return np.nan
    scores = []
    for sent in sentences:
        score = get_uncertainty_score(sent)
        if not np.isnan(score):
            scores.append(score)
    if not scores:
        return np.nan
    return float(np.mean(scores))

# ================================
# 读取 call 文本（与原代码逻辑一致）
# ================================
def load_person_labeled_sentences(person_file: Path):
    """读取带说话人标签的 CSV，返回说话最多的 speaker 的句子列表"""
    df = pd.read_csv(person_file)
    if "Person" in df.columns and "Sentence" in df.columns:
        speaker_counts = df.groupby("Person").size()
        if len(speaker_counts) > 0:
            ceo_person = speaker_counts.idxmax()
            sentences = df[df["Person"] == ceo_person]["Sentence"].tolist()
            return sentences
        else:
            return df["Sentence"].tolist()
    return []

def load_text_lines(text_file: Path):
    """读取纯文本文件，每行作为一句"""
    lines = text_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]

def read_call_text(call_dir: Path) -> str:
    """
    读取一个 call 的完整文本。
    优先使用 PERSON_LABEL_DIR 下的 text.csv（取说话最多的 speaker），
    否则回退到 call_dir/text.txt。
    """
    if PERSON_LABEL_DIR.exists():
        person_file = PERSON_LABEL_DIR / call_dir.name / "text.csv"
        if person_file.exists():
            sentences = load_person_labeled_sentences(person_file)
            if sentences:
                return " ".join(sentences)
    # 回退
    text_file = call_dir / "text.txt"
    if text_file.exists():
        sentences = load_text_lines(text_file)
        return " ".join(sentences)
    return ""

# ================================
# 主处理流程
# ================================
def main():
    if not MAEC_DATASET_DIR.exists():
        print(f"错误: 找不到 MAEC_Dataset 目录: {MAEC_DATASET_DIR.absolute()}")
        print("请确保当前工作目录中包含 MAEC_upstream/MAEC_Dataset 文件夹。")
        return

    call_dirs = sorted([p for p in MAEC_DATASET_DIR.iterdir() if p.is_dir()])
    results = []
    for call_dir in tqdm(call_dirs, desc="Processing calls"):
        call_id = call_dir.name
        text = read_call_text(call_dir)
        if not text or len(text.strip()) < 50:
            score = np.nan
        else:
            score = get_uncertainty_by_sentences(text)
        results.append({"call_id": call_id, "uncertainty_roberta": score})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"已保存到 {OUTPUT_CSV.absolute()}")
    print(df.describe())

if __name__ == "__main__":
    main()