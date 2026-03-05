"""
Climbing Route Recommendation System
====================================
Supervised neural-network-based binary classification: predict whether a user will like a route.
Uses NLP on route descriptions and structured metadata. Core model in PyTorch; sklearn for
preprocessing, metrics, and baselines.
"""

import os
import re
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42  # Controls all randomness so results are reproducible run‑to‑run
ROUTES_PATH = "MP_archive/mp_routes.csv"  # Path to the full Mountain Project routes export
USER_TICKS_PATH = "MP_personal/ticks.csv"  # Path to this user's personal ticks/ratings
LIKE_THRESHOLD = 3.5  # Minimum personal star rating that counts as "liked" when creating labels
MAX_VOCAB_SIZE = 8000  # Maximum number of distinct tokens to keep in the text vocabulary
MAX_SEQ_LEN = 128  # Maximum number of tokens per description (longer text is truncated)
PAD_IDX = 0  # Token ID reserved for padding shorter sequences
UNK_IDX = 1  # Token ID used for out‑of‑vocabulary / unknown tokens
BATCH_SIZE = 32  # Number of routes processed together in each training/eval step
EMBED_DIM = 64  # Size of the learned word embeddings for text tokens
HIDDEN_DIM = 64  # Size of hidden layers in the text encoder and fusion MLP
LEARNING_RATE = 1e-3  # Step size for the Adam optimizer during training
NUM_EPOCHS = 50  # Maximum number of passes over the training data
PATIENCE = 7  # Early‑stopping patience (stop if validation AUC doesn't improve for this many epochs)
PREPROC_DIR = "preproc_cache"  # Directory for caching preprocessed data between runs

# Cache paths for global pretraining so we don't have to rerun the 100k‑route
# training stage every time.
GLOBAL_MODEL_PATH = Path("route_recommender_global.pt")
GLOBAL_VOCAB_PATH = Path("route_recommender_vocab.json")

# Style keywords extracted from descriptions (multi-hot)
# Core styles plus additional angle/feature/hold/movement words from Project1.py
STYLE_KEYWORDS = [
    # angle / wall shape
    "slab", "vertical", "overhang", "roof", "face",
    # features
    "crack", "arete", "corner", "dihedral", "chimney", "flake",
    # holds
    "crimp", "jug", "pocket", "mono", "sloper", "undercling",
    # movement / feel
    "technical", "delicate", "balanc", "precise", "pump", "sustained", "boulder", "powerful", "reach", "runout"
]  # Climbing style words we detect in descriptions to build a style feature vector

# Route types for multi-hot encoding
ROUTE_TYPES = ["Sport", "Trad", "Boulder", "Aid", "Mixed", "Ice", "Alpine", "TR"]  # Canonical route type categories mapped into a multi‑hot type feature


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data Loading
# =============================================================================

def load_routes(path=ROUTES_PATH):
    """Load full Mountain Project routes dataset."""
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()
    return df


def load_user_ticks(path=USER_TICKS_PATH):
    """Load user-specific ticks with ratings."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def extract_route_id(url):
    """Extract numeric route ID from Mountain Project URL."""
    if pd.isna(url):
        return None
    m = re.search(r"/route/(\d+)/", str(url))
    return m.group(1) if m else None


def normalize_route_name(name):
    """Normalize route name for matching."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s)


# =============================================================================
# Preprocessing
# =============================================================================

def yds_to_ordinal(rating):
    """Convert YDS grade to continuous ordinal scale."""
    if pd.isna(rating):
        return np.nan
    s = str(rating).strip()
    m = re.search(r"V(\d+)", s, re.I)
    if m:
        return 20 + int(m.group(1))
    m = re.search(r"5\.(\d+)([abcd]?)([+-]?)(?:\s|$|,|/)", s)
    if not m:
        m = re.search(r"5\.(\d+)([abcd]?)([+-]?)", s)
    if not m:
        return np.nan
    base = int(m.group(1))
    letter = {"a": 0, "b": 0.25, "c": 0.5, "d": 0.75}.get((m.group(2) or "").lower(), 0)
    modifier = {"+": 0.15, "-": -0.15}.get(m.group(3) or "", 0)
    return base + letter + modifier


def build_multihot_styles(desc):
    """Extract style keywords from description (multi-hot)."""
    if pd.isna(desc):
        return [0] * len(STYLE_KEYWORDS)
    text = str(desc).lower()
    return [1 if kw in text else 0 for kw in STYLE_KEYWORDS]


def build_multihot_route_types(route_type_str):
    """Parse route type string into multi-hot vector (e.g., 'Sport, TR' -> [1,0,...,1])."""
    if pd.isna(route_type_str):
        return [0] * len(ROUTE_TYPES)
    parts = [p.strip() for p in str(route_type_str).split(",")]
    out = [0] * len(ROUTE_TYPES)
    for p in parts:
        for i, rt in enumerate(ROUTE_TYPES):
            if rt.lower() in p.lower() or p.lower() in rt.lower():
                out[i] = 1
                break
    return out


def merge_and_label(routes_df, ticks_df):
    """Merge user ticks with routes and create binary labels."""
    routes = routes_df.copy()
    ticks = ticks_df.copy()

    routes["route_id"] = routes["URL"].apply(extract_route_id)
    ticks["route_id"] = ticks["URL"].apply(extract_route_id)
    routes["route_norm"] = routes["Route"].apply(normalize_route_name)
    ticks["route_norm"] = ticks["Route"].apply(normalize_route_name)

    ticks["user_stars"] = pd.to_numeric(ticks["Your Stars"], errors="coerce").fillna(-1).astype(float)
    ticks["label"] = (ticks["user_stars"] >= LIKE_THRESHOLD).astype(float)

    route_cols = ["Route", "Location", "URL", "desc", "Rating", "Avg Stars", "num_votes", "Route Type", "Pitches", "Length"]
    route_cols = [c for c in route_cols if c in routes.columns]
    routes_sub = routes[["route_id", "route_norm"] + route_cols].copy()
    routes_sub.columns = ["route_id", "route_norm"] + [f"{c}_route" for c in route_cols]

    merged = ticks.merge(routes_sub, on="route_id", how="inner")
    if len(merged) == 0:
        merged = ticks.merge(routes_sub, left_on="route_norm", right_on="route_norm", how="inner")

    for c in route_cols:
        merged[c] = merged[f"{c}_route"]
        merged = merged.drop(columns=[f"{c}_route"], errors="ignore")
    return merged


def preprocess_data(routes_df, ticks_df, preproc_dir=PREPROC_DIR):
    """Full preprocessing pipeline."""
    os.makedirs(preproc_dir, exist_ok=True)

    merged = merge_and_label(routes_df, ticks_df)
    if len(merged) == 0:
        raise ValueError("No overlapping routes between ticks and full dataset. Check merge keys.")

    merged = merged.drop_duplicates(subset=["route_id"], keep="first")

    required = ["desc", "Rating", "Avg Stars", "num_votes", "Route Type"]
    for c in required:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged.dropna(subset=["desc", "Rating"])

    merged["grade_ord"] = merged["Rating"].apply(yds_to_ordinal)
    merged = merged.dropna(subset=["grade_ord"])

    merged["avg_stars"] = pd.to_numeric(merged["Avg Stars"], errors="coerce").fillna(0)
    merged["num_votes"] = pd.to_numeric(merged["num_votes"], errors="coerce").fillna(0).clip(lower=0)

    merged["style_vec"] = merged["desc"].apply(build_multihot_styles)
    merged["type_vec"] = merged["Route Type"].apply(build_multihot_route_types)

    # Normalize grade and average star features so the model trains more stably.
    g = merged["grade_ord"]
    merged["grade_norm"] = (g - g.mean()) / (g.std() + 1e-8)
    s = merged["avg_stars"]
    merged["stars_norm"] = (s - s.mean()) / (s.std() + 1e-8)

    return merged


# =============================================================================
# NLP Pipeline: turns route descriptions into numeric sequences
# =============================================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"~", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text):
    return clean_text(text).split()


def build_vocab(texts, max_vocab_size=MAX_VOCAB_SIZE):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for w, _ in counter.most_common(max_vocab_size - 2):
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def text_to_ids(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(t, UNK_IDX) for t in tokens]


def pad_or_truncate(ids, max_len, pad_val=PAD_IDX):
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_val] * (max_len - len(ids))


# =============================================================================
# Dataset Class: Turn preprocessed dataframe into model-ready tensors (text ids, features, label).
# =============================================================================

class RouteDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len=MAX_SEQ_LEN):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_ids = text_to_ids(row["desc"], self.vocab)
        if len(text_ids) == 0:
            text_ids = [PAD_IDX]
        token_ids = pad_or_truncate(text_ids, self.max_seq_len)
        length = min(len(text_ids), self.max_seq_len)

        struct = np.array(
            [row["grade_norm"], row["stars_norm"]]
            + list(row["style_vec"])
            + list(row["type_vec"]),
            dtype=np.float32,
        )
        struct = np.nan_to_num(struct, nan=0.0, posinf=0.0, neginf=0.0)

        label = float(row["label"])
        return {
            "token_ids": torch.LongTensor(token_ids),
            "length": torch.tensor(length, dtype=torch.long),
            "attention_mask": torch.tensor([1] * length + [0] * (self.max_seq_len - length), dtype=torch.float32),
            "structured_features": torch.FloatTensor(struct),
            "label": torch.FloatTensor([label]),
        }


def collate_fn(batch):
    return {
        "token_ids": torch.stack([b["token_ids"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "structured_features": torch.stack([b["structured_features"] for b in batch]),
        "label": torch.cat([b["label"] for b in batch]),
    }


# =============================================================================
# Model Architecture
# =============================================================================

class TextEncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)

    def forward(self, token_ids, lengths):
        x = self.embed(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]


class RouteRecommenderModel(nn.Module):
    def __init__(self, vocab_size, num_structured, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.text_encoder = TextEncoderLSTM(vocab_size, embed_dim, hidden_dim)
        text_out_dim = hidden_dim  # LSTM final hidden state
        struct_out_dim = hidden_dim  # struct_fc output

        self.struct_fc = nn.Sequential(
            nn.Linear(num_structured, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        fusion_in = text_out_dim + struct_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_ids, length, attention_mask, structured_features):
        text_repr = self.text_encoder(token_ids, length)
        struct_repr = self.struct_fc(structured_features)
        fused = torch.cat([text_repr, struct_repr], dim=1)
        return self.fusion(fused).squeeze(-1)


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        token_ids = batch["token_ids"].to(device)
        length = batch["length"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        struct = batch["structured_features"].to(device)
        label = batch["label"].to(device)
        logit = model(token_ids, length, attention_mask, struct)
        loss = criterion(logit, label)
        loss.backward()
        # clip_grad_norm_ especially necessary for LSTM training to avoid "exploding gradients". 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / (n_batches or 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            length = batch["length"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            struct = batch["structured_features"].to(device)
            label = batch["label"].to(device)
            logit = model(token_ids, length, attention_mask, struct)
            loss = criterion(logit, label)
            total_loss += loss.item()
            probs = torch.sigmoid(logit)
            preds.extend(probs.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return total_loss / len(loader), np.array(preds), np.array(labels)


def roc_auc(labels, preds):
    if np.sum(labels) == 0 or np.sum(labels) == len(labels):
        return 0.5
    return float(roc_auc_score(labels, preds))


def precision_recall_at_k(labels, preds, k=10):
    order = np.argsort(-preds)
    top_k = order[:k]
    relevant = np.sum(labels[top_k])
    return relevant / k, relevant / (np.sum(labels) + 1e-8)


def _learn_user_like_threshold(labels, preds, default_threshold=0.4, min_positives=3):
    """
    Learn a user-specific like threshold from validation data by maximizing F1.
    Falls back to default_threshold when there are too few positive examples.
    """
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds, dtype=np.float64)
    if labels.sum() < min_positives:
        return float(default_threshold)

    best_thr = float(default_threshold)
    best_f1 = -1.0
    # Scan a reasonable range of thresholds
    for thr in np.linspace(0.1, 0.9, 81):
        pred_pos = preds >= thr
        tp = np.logical_and(pred_pos, labels == 1).sum()
        fp = np.logical_and(pred_pos, labels == 0).sum()
        fn = np.logical_and(~pred_pos, labels == 1).sum()
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def train_model(model, train_loader, val_loader, device, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_auc = roc_auc(val_labels, val_preds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _build_structured_matrix(df):
    grade = np.asarray(df["grade_norm"].values, dtype=np.float64).reshape(-1, 1)
    stars = np.asarray(df["stars_norm"].values, dtype=np.float64).reshape(-1, 1)
    style = np.vstack(df["style_vec"].tolist())
    rtype = np.vstack(df["type_vec"].tolist())
    return np.hstack([grade, stars, style, rtype])


def popularity_baseline(routes_df, grade_band_width=0.5):
    df = routes_df.copy()
    df["grade_ord"] = df["Rating"].apply(yds_to_ordinal)
    df = df.dropna(subset=["grade_ord"])
    df["avg_stars"] = pd.to_numeric(df["Avg Stars"], errors="coerce").fillna(0)
    df["num_votes"] = pd.to_numeric(df["num_votes"], errors="coerce").fillna(0)
    return df


# =============================================================================
# Recommendation Functions
# =============================================================================

def preprocess_routes_only(routes_subset):
    r = routes_subset.copy()
    r["grade_ord"] = r["Rating"].apply(yds_to_ordinal)
    r = r.dropna(subset=["grade_ord", "desc"])
    r["avg_stars"] = pd.to_numeric(r["Avg Stars"], errors="coerce").fillna(0)
    r["num_votes"] = pd.to_numeric(r["num_votes"], errors="coerce").fillna(0).clip(lower=0)
    g = r["grade_ord"]
    r["grade_norm"] = (g - g.mean()) / (g.std() + 1e-8)
    s = r["avg_stars"]
    r["stars_norm"] = (s - s.mean()) / (s.std() + 1e-8)
    v = r["num_votes"].replace(0, 1)
    r["votes_norm"] = np.log1p(r["num_votes"].clip(0)) / (np.log1p(v).max() + 1e-8)
    r["style_vec"] = r["desc"].apply(build_multihot_styles)
    r["type_vec"] = r["Route Type"].apply(build_multihot_route_types)
    r["label"] = 0
    return r


def _grade_range_from_ticks(user_ticks_df, margin=1.0, floor_percentile=25):
    grades = user_ticks_df["Rating"].apply(yds_to_ordinal).dropna()
    if len(grades) == 0:
        return None, None
    lo = float(np.percentile(grades, floor_percentile)) - 0.25
    hi = float(grades.max()) + margin
    return max(5.0, lo), min(25.0, hi)


def _route_type_preferences_from_ticks(user_ticks_df):
    if "Route Type" not in user_ticks_df.columns:
        return None
    types = set()
    for val in user_ticks_df["Route Type"].dropna():
        for p in str(val).split(","):
            t = p.strip()
            if t:
                types.add(t)
    return types if types else None


def _route_type_preferences_from_liked(df):
    """
    Derive preferred route types only from routes the user actually liked
    (label == 1). This prevents recommending e.g. Trad if the user never
    rated Trad routes positively.
    """
    if "Route Type" not in df.columns or "label" not in df.columns:
        return None
    liked = df[df["label"] == 1.0]
    if liked.empty:
        return None
    types = set()
    for val in liked["Route Type"].dropna():
        for p in str(val).split(","):
            t = p.strip()
            if t:
                types.add(t)
    return types if types else None


def _pitch_range_from_ticks(user_ticks_df, margin=1):
    if "Pitches" not in user_ticks_df.columns:
        return None, None
    pitches = pd.to_numeric(user_ticks_df["Pitches"], errors="coerce").dropna()
    if len(pitches) == 0:
        return None, None
    lo = int(max(1, pitches.min() - margin))
    hi = int(pitches.max() + margin)
    return lo, hi


def _votes_confidence_multiplier(votes_norm, floor=0.5):
    return np.clip(floor + (1.0 - floor) * np.asarray(votes_norm, dtype=np.float64), 0.0, 1.0)


def _route_matches_type_preference(route_type_str, preferred_types):
    if not preferred_types:
        return True
    if pd.isna(route_type_str):
        return False
    route_types = {p.strip() for p in str(route_type_str).split(",") if p.strip()}
    return bool(route_types & preferred_types)


def generate_recommendations(
    model, routes_df, user_ticks_df, vocab, device,
    top_n=20, grade_min=None, grade_max=None, route_types=None, pitch_min=None, pitch_max=None, location_filter=None,
):
    routes_df = routes_df.copy()
    routes_df["route_id"] = routes_df["URL"].apply(extract_route_id)
    routes_df["route_norm"] = routes_df["Route"].apply(normalize_route_name)
    climbed_ids = set(user_ticks_df["URL"].apply(extract_route_id).dropna())
    climbed_names = set(user_ticks_df["Route"].apply(normalize_route_name))

    if grade_min is None or grade_max is None:
        gm, gx = _grade_range_from_ticks(user_ticks_df)
        if grade_min is None:
            grade_min = gm
        if grade_max is None:
            grade_max = gx
    if route_types is None:
        route_types = _route_type_preferences_from_ticks(user_ticks_df)
    if pitch_min is None or pitch_max is None:
        pm, px = _pitch_range_from_ticks(user_ticks_df)
        if pitch_min is None:
            pitch_min = pm
        if pitch_max is None:
            pitch_max = px

    unseen = routes_df[
        (~routes_df["route_id"].isin(climbed_ids)) & (~routes_df["route_norm"].isin(climbed_names))
    ].copy()
    unseen = preprocess_routes_only(unseen)
    if len(unseen) == 0:
        return []

    dataset = RouteDataset(unseen, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in loader:
            logit = model(
                batch["token_ids"].to(device),
                batch["length"].to(device),
                batch["attention_mask"].to(device),
                batch["structured_features"].to(device),
            )
            probs_list.extend(torch.sigmoid(logit).cpu().numpy())
    unseen = unseen.reset_index(drop=True)
    unseen["pred_prob"] = np.array(probs_list[: len(unseen)], dtype=np.float64)
    mult = _votes_confidence_multiplier(unseen["votes_norm"].values)
    unseen["pred_prob"] = unseen["pred_prob"].values * mult

    if grade_min is not None:
        unseen = unseen[unseen["grade_ord"] >= grade_min]
    if grade_max is not None:
        unseen = unseen[unseen["grade_ord"] <= grade_max]
    if route_types is not None:
        mask = unseen["Route Type"].apply(lambda x: _route_matches_type_preference(x, route_types))
        unseen = unseen[mask]
    if pitch_min is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts >= pitch_min]
    if pitch_max is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts <= pitch_max]
    if location_filter:
        unseen = unseen[unseen["Location"].astype(str).str.contains(location_filter, case=False, na=False)]

    recs = unseen.nlargest(top_n, "pred_prob")
    return recs[["Route", "Rating", "Location", "pred_prob"]].to_dict("records")


def generate_area_recommendations(
    model, routes_df, user_ticks_df, vocab, device,
    top_n=10, min_routes=3, like_threshold=0.4,
    grade_min=None, grade_max=None, route_types=None, pitch_min=None, pitch_max=None, location_filter=None,
):
    routes_df = routes_df.copy()
    routes_df["route_id"] = routes_df["URL"].apply(extract_route_id)
    routes_df["route_norm"] = routes_df["Route"].apply(normalize_route_name)
    climbed_ids = set(user_ticks_df["URL"].apply(extract_route_id).dropna())
    climbed_names = set(user_ticks_df["Route"].apply(normalize_route_name))

    if grade_min is None or grade_max is None:
        gm, gx = _grade_range_from_ticks(user_ticks_df)
        if grade_min is None:
            grade_min = gm
        if grade_max is None:
            grade_max = gx
    if route_types is None:
        route_types = _route_type_preferences_from_ticks(user_ticks_df)
    if pitch_min is None or pitch_max is None:
        pm, px = _pitch_range_from_ticks(user_ticks_df)
        if pitch_min is None:
            pitch_min = pm
        if pitch_max is None:
            pitch_max = px

    unseen = routes_df[
        (~routes_df["route_id"].isin(climbed_ids)) & (~routes_df["route_norm"].isin(climbed_names))
    ].copy()
    unseen = preprocess_routes_only(unseen)
    if len(unseen) == 0:
        return []

    dataset = RouteDataset(unseen, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in loader:
            logit = model(
                batch["token_ids"].to(device),
                batch["length"].to(device),
                batch["attention_mask"].to(device),
                batch["structured_features"].to(device),
            )
            probs_list.extend(torch.sigmoid(logit).cpu().numpy())
    unseen = unseen.reset_index(drop=True)
    unseen["pred_prob"] = np.array(probs_list[: len(unseen)], dtype=np.float64)
    mult = _votes_confidence_multiplier(unseen["votes_norm"].values)
    unseen["pred_prob"] = unseen["pred_prob"].values * mult

    if grade_min is not None:
        unseen = unseen[unseen["grade_ord"] >= grade_min]
    if grade_max is not None:
        unseen = unseen[unseen["grade_ord"] <= grade_max]
    if route_types is not None:
        mask = unseen["Route Type"].apply(lambda x: _route_matches_type_preference(x, route_types))
        unseen = unseen[mask]
    if pitch_min is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts >= pitch_min]
    if pitch_max is not None:
        pts = pd.to_numeric(unseen["Pitches"], errors="coerce")
        unseen = unseen[pts <= pitch_max]
    if location_filter:
        unseen = unseen[unseen["Location"].astype(str).str.contains(location_filter, case=False, na=False)]

    if len(unseen) == 0:
        return []

    grouped = _aggregate_areas_by_liked_count(unseen, like_threshold=like_threshold, min_routes=min_routes)
    if len(grouped) == 0:
        return []
    top_areas = grouped.head(top_n)
    return top_areas[
        ["Location", "liked_count", "liked_ratio", "score", "mean_prob", "total_routes", "Route", "Rating", "pred_prob"]
    ].rename(columns={"total_routes": "num_routes"}).to_dict("records")


def _aggregate_areas_by_liked_count(unseen, like_threshold=0.4, min_routes=3):
    unseen = unseen.copy()
    unseen["predicted_like"] = unseen["pred_prob"] >= like_threshold

    liked_count = unseen.groupby("Location")["predicted_like"].sum().astype(int)
    total_routes = unseen.groupby("Location").size()
    mean_prob = unseen.groupby("Location")["pred_prob"].mean()
    grouped = pd.DataFrame(
        {"liked_count": liked_count, "total_routes": total_routes, "mean_prob": mean_prob}
    ).reset_index()
    grouped["liked_ratio"] = grouped["liked_count"] / grouped["total_routes"].replace(0, np.nan)

    grouped = grouped[grouped["total_routes"] >= min_routes]
    if len(grouped) == 0:
        return grouped

    has_any_liked = grouped["liked_count"].max() >= 1
    if has_any_liked:
        grouped = grouped[grouped["liked_count"] >= 1]
        grouped["score"] = grouped["liked_count"] * grouped["liked_ratio"]
        sort_col = "score"
    else:
        grouped["score"] = grouped["mean_prob"]
        sort_col = "mean_prob"

    best_routes = (
        unseen.sort_values("pred_prob", ascending=False)
        .groupby("Location")
        .first()
        .reset_index()[["Location", "Route", "Rating", "pred_prob"]]
    )
    grouped = grouped.merge(best_routes, on="Location", how="left")

    return grouped.sort_values(sort_col, ascending=False)


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed()

    print("Loading data...")
    routes = load_routes()
    ticks = load_user_ticks()
    print(f"Routes: {len(routes)}, User ticks: {len(ticks)}")

    print("Preprocessing (personal ticks)...")
    df = preprocess_data(routes, ticks)
    print(f"Merged labeled samples (personal): {len(df)}")

    if len(df) < 10:
        print("Not enough labeled data. Need at least 10 samples.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Stage 1: Global pretraining (cached)
    # -----------------------------
    if GLOBAL_MODEL_PATH.exists() and GLOBAL_VOCAB_PATH.exists():
        print("\nLoading cached global model and vocabulary...")
        with GLOBAL_VOCAB_PATH.open("r") as f:
            vocab = json.load(f)
        num_struct = 2 + len(STYLE_KEYWORDS) + len(ROUTE_TYPES)
        model = RouteRecommenderModel(
            vocab_size=len(vocab),
            num_structured=num_struct,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
        state = torch.load(GLOBAL_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        print("Loaded global pretraining weights from disk.")
    else:
        # Build vocabulary from all route descriptions so embeddings see global text
        vocab = build_vocab(routes["desc"].astype(str).tolist(), MAX_VOCAB_SIZE)
        print(f"Vocab size: {len(vocab)}")

        print("\nPretraining on global routes using Avg Stars...")
        global_df = preprocess_routes_only(routes)
        # Use binary label: high vs low average stars
        global_df = global_df.copy()
        global_df["label"] = (global_df["avg_stars"] >= LIKE_THRESHOLD).astype(float)

        g_idx = np.random.permutation(len(global_df))
        g_split = int(0.9 * len(global_df))
        global_train_df = global_df.iloc[g_idx[:g_split]]
        global_val_df = global_df.iloc[g_idx[g_split:]]

        global_train_ds = RouteDataset(global_train_df.reset_index(drop=True), vocab, MAX_SEQ_LEN)
        global_val_ds = RouteDataset(global_val_df.reset_index(drop=True), vocab, MAX_SEQ_LEN)
        global_train_loader = DataLoader(
            global_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        global_val_loader = DataLoader(
            global_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        num_struct = 2 + len(STYLE_KEYWORDS) + len(ROUTE_TYPES)
        model = RouteRecommenderModel(
            vocab_size=len(vocab),
            num_structured=num_struct,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        ).to(device)

        # Pretrain model to predict globally popular routes (small number of epochs, default LR)
        _ = train_model(
            model,
            global_train_loader,
            global_val_loader,
            device,
            learning_rate=LEARNING_RATE,
            num_epochs=5,
        )

        # Cache global pretraining weights + vocab for future runs.
        try:
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            with GLOBAL_VOCAB_PATH.open("w") as f:
                json.dump(vocab, f)
            print(f"Saved global model to {GLOBAL_MODEL_PATH} and vocab to {GLOBAL_VOCAB_PATH}")
        except Exception as e:
            print(f"Warning: failed to save global model/vocab: {e}")

    # -----------------------------
    # Stage 2: Finetune on personal like/dislike labels
    # -----------------------------
    print("\nFinetuning on personal ticks...")
    idx = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    train_df = df.iloc[idx[:split]]
    val_df = df.iloc[idx[split:]]

    # Freeze lower layers: keep global text + structured encoders, adapt only fusion head
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    for p in model.struct_fc.parameters():
        p.requires_grad = False

    train_ds = RouteDataset(train_df.reset_index(drop=True), vocab, MAX_SEQ_LEN)
    val_ds = RouteDataset(val_df.reset_index(drop=True), vocab, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Finetune with smaller learning rate; NUM_EPOCHS and patience still control early stopping
    history = train_model(model, train_loader, val_loader, device, learning_rate=LEARNING_RATE * 0.1)

    HISTORY_PATH = Path("history.json")
    CHECKPOINT_PATH = Path("route_recommender_checkpoint.pt")
    try:
        with HISTORY_PATH.open("w") as f:
            json.dump(history, f)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved history to {HISTORY_PATH} and checkpoint to {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Warning: failed to save history/checkpoint: {e}")

    print("\nEvaluation on validation set:")
    _, val_preds, val_labels = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), device)
    auc = roc_auc(val_labels, val_preds)
    p5, r5 = precision_recall_at_k(val_labels, val_preds, k=5)
    p10, r10 = precision_recall_at_k(val_labels, val_preds, k=10)
    user_like_threshold = _learn_user_like_threshold(val_labels, val_preds, default_threshold=0.4)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Precision@5: {p5:.4f}, Recall@5: {r5:.4f}")
    print(f"Precision@10: {p10:.4f}, Recall@10: {r10:.4f}")
    print(f"Learned user like-threshold: {user_like_threshold:.3f}")

    if "avg_stars" in val_df.columns:
        pop_scores = val_df["avg_stars"].values
    elif "Avg Stars" in val_df.columns:
        pop_scores = pd.to_numeric(val_df["Avg Stars"], errors="coerce").fillna(0).values
    else:
        pop_scores = np.zeros_like(val_labels)
    pop_auc = roc_auc(val_labels, pop_scores[: len(val_labels)])
    print(f"\nPopularity baseline ROC-AUC: {pop_auc:.4f}")

    X_train = _build_structured_matrix(train_df)
    X_val = _build_structured_matrix(val_df)
    y_train, y_val = train_df["label"].values, val_df["label"].values
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc(y_val, lr_probs)
    print(f"LogisticRegression baseline ROC-AUC: {lr_auc:.4f}")

    # Restrict recommendations to route types the user has actually *ticked* at least once.
    # If you've never logged a given type (e.g. Trad), it won't be recommended.
    preferred_types = _route_type_preferences_from_ticks(ticks)

    print("\nTop 10 recommendations (unseen routes):")
    recs = generate_recommendations(
        model, routes, ticks, vocab, device, top_n=10, route_types=preferred_types
    )
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['Route']} ({r['Rating']}) - prob={r['pred_prob']:.3f} - {r['Location'][:50]}...")

    print("\nTop 10 recommended areas (score = liked_count × liked_ratio):")
    area_recs = generate_area_recommendations(
        model,
        routes,
        ticks,
        vocab,
        device,
        top_n=10,
        like_threshold=user_like_threshold,
        route_types=preferred_types,
    )
    for i, a in enumerate(area_recs, 1):
        print(
            f"  {i}. {a['Location']} | liked={int(a['liked_count'])}/{int(a['num_routes'])} "
            f"(ratio={a['liked_ratio']:.2f}) score={a['score']:.2f} | top route: {a['Route']} ({a['Rating']}) "
            f"prob={a['pred_prob']:.3f}"
        )

    return model, vocab, history


if __name__ == "__main__":
    main()

