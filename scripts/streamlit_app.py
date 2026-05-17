import hashlib
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Signals", layout="wide")

APP_DIR = Path(__file__).resolve().parent
# Works whether this file sits at the repo root or inside an app/ folder.
REPO_ROOT = APP_DIR if (APP_DIR / "data").exists() else APP_DIR.parent
CSV_PATH = REPO_ROOT / "data/processed/processed_signals.csv"
IMAGE_BASE_URL = os.getenv("SIGNALS_IMAGE_BASE_URL", "").rstrip("/")


# -----------------------------
# Helpers
# -----------------------------
def pick_column(df, candidates):
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lowered:
            return lowered[key]
    return None



def safe_text(val):
    if pd.isna(val):
        return "NA"
    text = str(val).strip()
    return text if text else "NA"



def normalize_text(val):
    if pd.isna(val):
        return ""
    return str(val).strip()



def extract_hashtags(text, lower=True):
    if not text or pd.isna(text):
        return []
    tags = re.findall(r"#[A-Za-z0-9_\-/]+", str(text))
    return [t.lower() if lower else t for t in tags]



def build_search_text(row, cols):
    parts = []
    for c in cols:
        if c and c in row.index:
            v = normalize_text(row[c])
            if v and v != "NA":
                parts.append(v)
    return " | ".join(parts)



def resolve_local_file(candidate: str):
    if not candidate or candidate == "NA":
        return None

    path = Path(candidate)
    probes = []
    if path.is_absolute():
        probes.append(path)
    else:
        probes.extend([
            Path.cwd() / path,
            REPO_ROOT / path,
            APP_DIR / path,
        ])

    for probe in probes:
        if probe.exists():
            return probe
    return None



def get_image_candidates(row, col_link, col_image):
    candidates = []

    if col_image and col_image in row.index:
        image_val = safe_text(row[col_image])
        if image_val != "NA":
            candidates.append(image_val)
            if IMAGE_BASE_URL and not image_val.lower().startswith(("http://", "https://")):
                candidates.append(f"{IMAGE_BASE_URL}/{image_val.lstrip('./')}")

    if col_link and col_link in row.index:
        link_val = safe_text(row[col_link])
        if link_val != "NA":
            candidates.append(link_val)

    return candidates



def show_image_from_candidates(candidates):
    shown = False
    for cand in candidates:
        resolved = resolve_local_file(cand)
        if resolved:
            st.image(str(resolved), use_container_width=True)
            shown = True
            break

        if cand.lower().startswith(("http://", "https://")) and any(
            cand.lower().split("?")[0].endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]
        ):
            st.image(cand, use_container_width=True)
            shown = True
            break

    return shown


def short_text(val, max_chars=220):
    text = safe_text(val)
    if text == "NA":
        return "NA"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def normalize_hashtag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag:
        return ""
    tag = tag if tag.startswith("#") else f"#{tag}"
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"[^#A-Za-z0-9_\-/]+", "", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag if len(tag) > 1 else ""


def stable_colour(text: str) -> str:
    palette = [
        "#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5", "#E0F2F1",
        "#FCE4EC", "#EDE7F6", "#F1F8E9", "#FFFDE7", "#ECEFF1",
    ]
    digest = int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8], 16)
    return palette[digest % len(palette)]


# Domain colours for the visible signal domains.
# Use these for the main domain badge.
DOMAIN_COLOURS = {
    "TECH": "#1E88E5",              # blue
    "SOCIETY_DIGITAL": "#FFF59D",   # light yellow
    "SOCIETY_HEALTH": "#F9A825",    # dark yellow
    "ENVIRONMENT": "#43A047",       # green
    "POLITICS": "#8E24AA",          # purple
    "ECONOMY": "#E53935",           # red
    "SECURITY": "#9E9E9E",          # grey
    "OTHERS": "#F06292",            # pink
}

# Flexible matching, because CSV labels may vary slightly.
DOMAIN_ALIASES = {
    "TECH": "TECH",
    "TECHNOLOGY": "TECH",
    "DIGITAL": "TECH",
    "AI": "TECH",
    "SOCIETY DIGITAL": "SOCIETY_DIGITAL",
    "SOCIETY - DIGITAL": "SOCIETY_DIGITAL",
    "SOCIETY_DIGITAL": "SOCIETY_DIGITAL",
    "DIGITAL SOCIETY": "SOCIETY_DIGITAL",
    "SOCIETY HEALTH": "SOCIETY_HEALTH",
    "SOCIETY - HEALTH": "SOCIETY_HEALTH",
    "SOCIETY_HEALTH": "SOCIETY_HEALTH",
    "HEALTH": "SOCIETY_HEALTH",
    "ENVIRONMENT": "ENVIRONMENT",
    "CLIMATE": "ENVIRONMENT",
    "POLITICS": "POLITICS",
    "POLITICAL": "POLITICS",
    "GOVERNANCE": "POLITICS",
    "ECONOMY": "ECONOMY",
    "ECONOMIC": "ECONOMY",
    "BUSINESS": "ECONOMY",
    "FINANCE": "ECONOMY",
    "SECURITY": "SECURITY",
    "DEFENCE": "SECURITY",
    "DEFENSE": "SECURITY",
    "OTHERS": "OTHERS",
    "OTHER": "OTHERS",
    "MISC": "OTHERS",
}

DOMAIN_LABELS = {
    "TECH": "TECH, Science, Frontiers",
    "SOCIETY_DIGITAL": "SOCIETY: Digital, Culture, Psychology",
    "SOCIETY_HEALTH": "SOCIETY: Health, Augmentation, Demographics",
    "ENVIRONMENT": "ENVIRONMENT, Infra, Energy",
    "POLITICS": "POLITICS, Governance, Power",
    "ECONOMY": "ECONOMY, Jobs, Learning",
    "SECURITY": "SECURITY, Military, Grey ops",
    "OTHERS": "OTHERS",
}

# Optional: map recurring tags to a domain, so tags can take their own domain hue
# instead of always inheriting the signal's domain. Add your team's tags here.
TAG_DOMAIN_MAP = {
    "#K": "ECONOMY",
    "#WATER": "ENVIRONMENT",
    "#AI": "TECH",
    "#CLIMATE": "ENVIRONMENT",
    "#HEALTH": "SOCIETY_HEALTH",
    "#AGEING": "SOCIETY_HEALTH",
    "#AGING": "SOCIETY_HEALTH",
    "#GEOPOLITICS": "POLITICS",
    "#SECURITY": "SECURITY",
}

TAG_DOMAIN_KEYWORDS = {
    "TECH": ["ai", "tech", "digital", "robot", "compute", "cyber", "data", "platform"],
    "SOCIETY_DIGITAL": ["digital", "social", "youth", "education", "media", "identity"],
    "SOCIETY_HEALTH": ["health", "ageing", "aging", "care", "mental", "disease", "hospital"],
    "ENVIRONMENT": ["climate", "water", "energy", "food", "carbon", "green", "biodiversity"],
    "POLITICS": ["politic", "governance", "election", "state", "policy", "geopolitic"],
    "ECONOMY": ["econom", "finance", "market", "trade", "job", "work", "labour", "labor", "k"],
    "SECURITY": ["security", "defence", "defense", "war", "conflict", "military", "crime"],
}

# Lighter tints for tag bubbles. The main domain badge uses the stronger base colour.
TAG_TINTS = {
    "TECH": "#BBDEFB",
    "SOCIETY_DIGITAL": "#FFF9C4",
    "SOCIETY_HEALTH": "#FFE082",
    "ENVIRONMENT": "#C8E6C9",
    "POLITICS": "#E1BEE7",
    "ECONOMY": "#FFCDD2",
    "SECURITY": "#E0E0E0",
    "OTHERS": "#F8BBD0",
}


def normalise_domain(value: str) -> str:
    """Return the canonical domain key used by the colour map."""
    text = safe_text(value)
    if text == "NA":
        return "OTHERS"

    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    if not cleaned:
        return "OTHERS"

    # The WhatsApp group/channel names may appear either as the original long
    # names or as older short labels. Check the distinctive phrases first so
    # SOCIETY: Digital does not get misread as TECH just because it contains
    # the word "Digital".
    if "SOCIETY" in cleaned and any(k in cleaned for k in ["HEALTH", "AUGMENTATION", "DEMOGRAPHICS"]):
        return "SOCIETY_HEALTH"
    if "SOCIETY" in cleaned and any(k in cleaned for k in ["DIGITAL", "CULTURE", "PSYCHOLOGY"]):
        return "SOCIETY_DIGITAL"
    if any(k in cleaned for k in ["TECH", "SCIENCE", "FRONTIERS", "TECHNOLOGY"]):
        return "TECH"
    if any(k in cleaned for k in ["ENVIRONMENT", "INFRA", "ENERGY", "CLIMATE"]):
        return "ENVIRONMENT"
    if any(k in cleaned for k in ["POLITICS", "GOVERNANCE", "POWER", "POLITICAL"]):
        return "POLITICS"
    if any(k in cleaned for k in ["ECONOMY", "JOBS", "LEARNING", "ECONOMIC", "BUSINESS", "FINANCE"]):
        return "ECONOMY"
    if any(k in cleaned for k in ["SECURITY", "MILITARY", "GREY", "DEFENCE", "DEFENSE"]):
        return "SECURITY"
    if any(k in cleaned for k in ["OTHERS", "OTHER", "MISC"]):
        return "OTHERS"

    if cleaned in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[cleaned]
    return "OTHERS"


def display_channel_label(value: str) -> str:
    """Show the original WhatsApp group/channel naming style in the UI."""
    return DOMAIN_LABELS.get(normalise_domain(value), "OTHERS")


def text_colour_for_background(hex_colour: str) -> str:
    """Pick black/white text based on simple perceived brightness."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = tuple(int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#111" if brightness > 165 else "#fff"


def chip_html(label, bg, fg=None, bold=False):
    fg = fg or text_colour_for_background(bg)
    weight = "600" if bold else "500"
    return (
        f'<span style="display:inline-block; padding:0.22rem 0.55rem; margin:0.12rem; '
        f'border-radius:999px; background:{bg}; color:{fg}; font-size:0.82rem; '
        f'font-weight:{weight}; border:1px solid rgba(0,0,0,0.08);">{label}</span>'
    )


def render_domain_chip(domain_value):
    canonical = normalise_domain(domain_value)
    label = display_channel_label(domain_value)
    colour = DOMAIN_COLOURS.get(canonical, DOMAIN_COLOURS["OTHERS"])
    st.markdown(chip_html(label, colour, bold=True), unsafe_allow_html=True)


def guess_tag_domain(tag: str, fallback_domain="OTHERS") -> str:
    tag = normalize_hashtag(tag)
    if not tag:
        return fallback_domain
    upper_tag = tag.upper()
    if upper_tag in TAG_DOMAIN_MAP:
        return TAG_DOMAIN_MAP[upper_tag]

    plain = tag.lstrip("#").replace("_", "-").lower()
    for domain, keywords in TAG_DOMAIN_KEYWORDS.items():
        if any(keyword in plain for keyword in keywords):
            return domain
    return fallback_domain


def render_tag_bubbles(tags, signal_domain="OTHERS", max_items=12, domain_aware=True):
    clean_tags = [str(tag).strip() for tag in tags if str(tag).strip() and str(tag).strip() != "NA"]
    if not clean_tags:
        return
    fallback = normalise_domain(signal_domain)
    chips = []
    for tag in clean_tags[:max_items]:
        tag_domain = guess_tag_domain(tag, fallback_domain=fallback) if domain_aware else fallback
        bg = TAG_TINTS.get(tag_domain, TAG_TINTS["OTHERS"])
        chips.append(chip_html(tag, bg, fg="#222"))
    st.markdown("".join(chips), unsafe_allow_html=True)


def render_bubbles(items, max_items=12):
    clean_items = [str(item).strip() for item in items if str(item).strip() and str(item).strip() != "NA"]
    if not clean_items:
        return
    chips = []
    for item in clean_items[:max_items]:
        bg = stable_colour(item.lower())
        chips.append(
            f'<span style="display:inline-block; padding:0.22rem 0.55rem; margin:0.12rem; '
            f'border-radius:999px; background:{bg}; color:#222; font-size:0.82rem; '
            f'border:1px solid rgba(0,0,0,0.06);">{item}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


def channel_label(value: str) -> str:
    text = safe_text(value)
    if text == "NA":
        return "NA"
    return text.upper()


def build_combined_hashtags(row) -> list:
    tags = []
    if col_tags and col_tags in row.index:
        tags.extend(extract_hashtags(row.get(col_tags), lower=False))
    if "user_added_hashtags" in row.index:
        tags.extend(extract_hashtags(row.get("user_added_hashtags"), lower=False))
    deduped = []
    seen = set()
    for tag in tags:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tag)
    return deduped


def render_signal_card(row, idx, semantic_query=""):
    """Render one signal as a compact card for the grid layout."""
    with st.container(border=True):
        asset_type = safe_text(row.get(col_type))
        link = safe_text(row.get(col_link)) if col_link else "NA"

        shown = show_image_from_candidates(get_image_candidates(row, col_link, col_image))
        if not shown and asset_type.lower() == "image":
            st.caption("Image not found in deployment")

        header = safe_text(row.get(col_header)) if col_header else "NA"
        if header == "NA" and asset_type.lower() == "image":
            header = "Image signal"

        st.markdown(f"#### {header}")

        # Topic domain: currently read from sub_channel_name / channel.
        # Source domain below remains the website domain, e.g. ft.com or bloomberg.com.
        signal_domain = safe_text(row.get(col_channel)) if col_channel else "OTHERS"
        if signal_domain != "NA":
            render_domain_chip(signal_domain)

        meta = []
        if col_domain:
            source_domain = safe_text(row.get(col_domain))
            if source_domain != "NA":
                meta.append(source_domain)
        if col_time:
            time_val = safe_text(row.get(col_time))
            if time_val != "NA":
                meta.append(time_val)
        if meta:
            st.caption(" · ".join(meta))

        if col_summary:
            summary = short_text(row.get(col_summary), 260)
            if summary != "NA":
                st.markdown(summary)

        parsed = build_combined_hashtags(row)
        if parsed:
            render_tag_bubbles(parsed[:8], signal_domain=signal_domain, domain_aware=True)
        else:
            render_bubbles(["No hashtag (N/A)"], max_items=1)

        signal_id = safe_text(row.get(col_id)) if col_id else str(idx)
        with st.expander("Add hashtag", expanded=False):
            new_tags = st.text_input(
                "Add one or more hashtags",
                key=widget_key("add_tag", signal_id),
                placeholder="#Economy #AI or Economy, AI",
            )
            if st.button("Save hashtag", key=widget_key("save_tag", signal_id)):
                saved = save_user_hashtags(signal_id, new_tags)
                if saved:
                    st.success(f"Saved: {' '.join(saved)}")
                    st.rerun()
                else:
                    st.info("Add at least one valid hashtag before saving.")

        upvotes = int(row.get("upvotes", 0))
        downvotes = int(row.get("downvotes", 0))
        notes = int(row.get("notes", 0))
        veto_label = " | Vetoed" if downvotes > 0 else ""

        with st.expander(f"Your Opinion: 👍 {upvotes} | Notes {notes}{veto_label}", expanded=False):
            vote_col1, vote_col2 = st.columns(2)
            with vote_col1:
                if st.button("👍 Useful / emerging", key=widget_key("up", signal_id)):
                    save_vote(signal_id, "up")
                    st.success("Vote saved.")
                    st.rerun()
            with vote_col2:
                if st.button("👎 Not useful", key=widget_key("down", signal_id)):
                    save_vote(signal_id, "down")
                    st.warning("Vote saved.")
                    st.rerun()

            comment = st.text_input(
                "Optional note",
                key=widget_key("comment", signal_id),
                placeholder="Why is this useful, emerging, noisy, or irrelevant?",
            )
            if st.button("Save note", key=widget_key("note", signal_id)):
                if comment.strip():
                    save_note(signal_id, comment)
                    st.success("Note saved.")
                    st.rerun()
                else:
                    st.info("Write a note before saving.")

        if semantic_query.strip() and "semantic_score" in df.columns:
            score = df.loc[idx, "semantic_score"]
            if pd.notna(score):
                st.caption(f"Semantic similarity: {score:.3f}")

        if link != "NA":
            st.markdown(f"[Open original]({link})")


# -----------------------------
# Supabase human review / voting helpers
# -----------------------------
VOTE_COLUMNS = ["signal_id", "vote", "comment", "timestamp"]
USER_TAG_COLUMNS = ["signal_id", "hashtag", "timestamp"]


@st.cache_resource
def get_supabase():
    """Create one cached Supabase client for the Streamlit app."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def widget_key(prefix, value):
    """Make short, stable Streamlit widget keys from long IDs/URLs."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _empty_df(columns):
    return pd.DataFrame(columns=columns)


def load_user_hashtags():
    """Load human-added hashtags from Supabase."""
    try:
        supabase = get_supabase()
        result = supabase.table("signal_hashtags").select("*").execute()
        data = result.data or []
    except Exception as exc:
        st.warning(f"Could not load Supabase hashtags: {exc}")
        return _empty_df(USER_TAG_COLUMNS)

    tags = pd.DataFrame(data)
    if tags.empty:
        return _empty_df(USER_TAG_COLUMNS)

    # Supabase stores created_at; the old app expected timestamp.
    if "timestamp" not in tags.columns and "created_at" in tags.columns:
        tags["timestamp"] = tags["created_at"]

    for col in USER_TAG_COLUMNS:
        if col not in tags.columns:
            tags[col] = ""

    tags["signal_id"] = tags["signal_id"].astype(str)
    tags["hashtag"] = tags["hashtag"].fillna("").apply(normalize_hashtag)
    tags = tags[tags["hashtag"] != ""]
    return tags[USER_TAG_COLUMNS]


# Backwards-compatible name used elsewhere in the app.
def load_user_tags():
    return load_user_hashtags()


def save_user_hashtag(signal_id, hashtag):
    """Save one normalized hashtag to Supabase."""
    hashtag = normalize_hashtag(hashtag)
    if not hashtag:
        return False

    try:
        supabase = get_supabase()
        supabase.table("signal_hashtags").insert({
            "signal_id": str(signal_id),
            "hashtag": hashtag,
        }).execute()
        return True
    except Exception as exc:
        st.error(f"Could not save hashtag to Supabase: {exc}")
        return False


def save_user_hashtags(signal_id, raw_tags):
    """Save one or more hashtags from a text input to Supabase."""
    candidates = re.split(r"[,\s]+", str(raw_tags or ""))
    cleaned = []
    seen = set()
    for candidate in candidates:
        tag = normalize_hashtag(candidate)
        if tag and tag.lower() not in seen:
            cleaned.append(tag)
            seen.add(tag.lower())

    if not cleaned:
        return []

    saved = []
    existing = load_user_hashtags()
    existing_keys = (
        set(zip(existing["signal_id"].astype(str), existing["hashtag"].str.lower()))
        if not existing.empty
        else set()
    )

    for tag in cleaned:
        key = (str(signal_id), tag.lower())
        if key in existing_keys:
            saved.append(tag)
            continue
        if save_user_hashtag(signal_id, tag):
            saved.append(tag)

    return saved


def user_tag_summary():
    tags = load_user_hashtags()
    if tags.empty:
        return pd.DataFrame(columns=["signal_id", "user_added_hashtags"])

    return (
        tags.groupby("signal_id")["hashtag"]
        .apply(lambda vals: " ".join(dict.fromkeys(vals)))
        .reset_index(name="user_added_hashtags")
    )


def save_vote(signal_id, vote_type):
    """Save an up/down vote to Supabase."""
    if vote_type not in ["up", "down"]:
        return False

    try:
        supabase = get_supabase()
        supabase.table("signal_votes").insert({
            "signal_id": str(signal_id),
            "vote_type": vote_type,
        }).execute()
        return True
    except Exception as exc:
        st.error(f"Could not save vote to Supabase: {exc}")
        return False


def save_note(signal_id, note):
    """Save an optional note to the separate signal_notes table."""
    note = str(note or "").strip()
    if not note:
        return False

    try:
        supabase = get_supabase()
        supabase.table("signal_notes").insert({
            "signal_id": str(signal_id),
            "note": note,
        }).execute()
        return True
    except Exception as exc:
        st.error(f"Could not save note to Supabase: {exc}")
        return False


def load_votes():
    """Load votes and notes from Supabase."""
    try:
        supabase = get_supabase()
        result = supabase.table("signal_votes").select("*").execute()
        data = result.data or []
    except Exception as exc:
        st.warning(f"Could not load Supabase votes: {exc}")
        return _empty_df(VOTE_COLUMNS)

    votes = pd.DataFrame(data)
    if votes.empty:
        return _empty_df(VOTE_COLUMNS)

    # Supabase schema uses vote_type; the old app expected vote.
    if "vote" not in votes.columns and "vote_type" in votes.columns:
        votes["vote"] = votes["vote_type"]
    if "timestamp" not in votes.columns and "created_at" in votes.columns:
        votes["timestamp"] = votes["created_at"]
    if "comment" not in votes.columns:
        votes["comment"] = ""

    for col in VOTE_COLUMNS:
        if col not in votes.columns:
            votes[col] = ""

    votes["signal_id"] = votes["signal_id"].astype(str)
    votes["vote"] = votes["vote"].fillna("").astype(str)
    return votes[VOTE_COLUMNS]


def load_notes():
    """Load notes from signal_notes if you decide to use the separate notes table."""
    try:
        supabase = get_supabase()
        result = supabase.table("signal_notes").select("*").execute()
        return pd.DataFrame(result.data or [])
    except Exception:
        return pd.DataFrame(columns=["signal_id", "note", "created_at"])


def vote_summary():
    votes = load_votes()
    notes_df = load_notes()

    if votes.empty:
        vote_counts = pd.DataFrame(columns=["signal_id", "upvotes", "downvotes", "score"])
    else:
        votes["signal_id"] = votes["signal_id"].astype(str)
        vote_counts = votes.pivot_table(
            index="signal_id",
            columns="vote",
            aggfunc="size",
            fill_value=0,
        ).reset_index()

        for col in ["up", "down"]:
            if col not in vote_counts.columns:
                vote_counts[col] = 0

        vote_counts["upvotes"] = vote_counts["up"].astype(int)
        vote_counts["downvotes"] = vote_counts["down"].astype(int)
        vote_counts["score"] = vote_counts["upvotes"] - vote_counts["downvotes"]
        vote_counts = vote_counts[["signal_id", "upvotes", "downvotes", "score"]]

    if notes_df.empty or "signal_id" not in notes_df.columns:
        note_counts = pd.DataFrame(columns=["signal_id", "notes"])
    else:
        notes_df["signal_id"] = notes_df["signal_id"].astype(str)
        note_counts = notes_df.groupby("signal_id").size().reset_index(name="notes")

    if vote_counts.empty and note_counts.empty:
        return pd.DataFrame(columns=["signal_id", "upvotes", "downvotes", "notes", "score"])

    summary = vote_counts.merge(note_counts, on="signal_id", how="outer")
    for col in ["upvotes", "downvotes", "notes", "score"]:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    return summary[["signal_id", "upvotes", "downvotes", "notes", "score"]]

def count_frame(frame, column, label_name, top_n=15, include_na=False):
    if not column or column not in frame.columns:
        return pd.DataFrame(columns=[label_name, "count"])
    series = frame[column].fillna("NA").astype(str)
    if not include_na:
        series = series.replace("NA", pd.NA).dropna()
    counts = series.value_counts().head(top_n).rename_axis(label_name).reset_index(name="count")
    return counts.sort_values("count", ascending=False)


def render_sorted_bar_chart(counts_df, label_col, count_col="count"):
    """Render bars sorted from largest to smallest, top to bottom."""
    if counts_df.empty:
        return
    ordered = counts_df.sort_values(count_col, ascending=False).reset_index(drop=True)
    chart = (
        alt.Chart(ordered)
        .mark_bar()
        .encode(
            x=alt.X(f"{count_col}:Q", title="Count"),
            y=alt.Y(f"{label_col}:N", sort="-x", title=None),
            tooltip=[alt.Tooltip(f"{label_col}:N", title=label_col), alt.Tooltip(f"{count_col}:Q", title="count")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        return None
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=False)



def add_cluster_labels(df, embedding_matrix, n_clusters=8):
    if len(df) < 3:
        df["cluster_id"] = "Cluster 1"
        return df

    k = min(n_clusters, len(df))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embedding_matrix)
    df["cluster_id"] = [f"Cluster {i+1}" for i in labels]
    return df



def label_clusters(df, col_tags, col_header):
    cluster_names = {}
    for cluster in sorted(df["cluster_id"].unique()):
        subset = df[df["cluster_id"] == cluster]

        tag_counter = Counter()
        if col_tags:
            for val in subset[col_tags].fillna(""):
                tag_counter.update(extract_hashtags(val))

        if tag_counter:
            top_tags = [t for t, _ in tag_counter.most_common(2)]
            cluster_names[cluster] = " / ".join(top_tags)
        else:
            headers = subset[col_header].fillna("").astype(str).tolist() if col_header else []
            label = headers[0][:50] if headers else cluster
            cluster_names[cluster] = label

    df["cluster_label"] = df["cluster_id"].map(cluster_names)
    return df





def flatten_tags(series):
    counter = Counter()
    for val in series.fillna(""):
        counter.update(extract_hashtags(val))
    return counter


def hashtag_pair_frame(frame, top_n=5):
    """Count records where two hashtags appear together.

    Each row contributes at most once to a given pair, even if the same
    hashtag appears multiple times in that row. The count is therefore
    number of matching records containing both tags, not raw tag mentions.
    """
    pair_counter = Counter()

    if "parsed_hashtags" not in frame.columns:
        return pd.DataFrame(columns=["hashtag_pair", "records_together"])

    for tags in frame["parsed_hashtags"]:
        unique_tags = []
        seen = set()
        for tag in tags:
            clean_tag = normalize_hashtag(tag)
            key = clean_tag.lower()
            if clean_tag and key not in seen:
                seen.add(key)
                unique_tags.append(clean_tag)

        unique_tags = sorted(unique_tags, key=str.lower)
        if len(unique_tags) < 2:
            continue

        for i in range(len(unique_tags)):
            for j in range(i + 1, len(unique_tags)):
                pair_counter[(unique_tags[i], unique_tags[j])] += 1

    if not pair_counter:
        return pd.DataFrame(columns=["hashtag_pair", "records_together"])

    rows = [
        {"hashtag_pair": f"{a} + {b}", "records_together": count}
        for (a, b), count in pair_counter.most_common(top_n)
    ]
    return pd.DataFrame(rows)


# -----------------------------
# Load data
# -----------------------------
df = load_data()
if df is None:
    st.error(f"Could not find {CSV_PATH}")
    st.stop()

# Flexible column matching
col_id = pick_column(df, ["signal_id", "record_id"])
if col_id is None:
    def make_signal_id(row):
        raw = str(row.get("final_url", "") or row.get("link_url", "") or row.get("link/image", "") or row.get("scraped_header", ""))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    df["signal_id"] = df.apply(make_signal_id, axis=1)
    col_id = "signal_id"
col_time = pick_column(df, ["message_time", "time of message"])
col_type = pick_column(df, ["asset_type", "asset type"])
col_link = pick_column(df, ["final_url", "link_url", "link/image"])
col_image = pick_column(df, ["image_path", "image file", "local_image_path"])
col_header = pick_column(df, ["scraped_header", "scraped header of the link"])
col_channel = pick_column(df, ["sub_channel_name", "sub channel name"])
col_summary = None
col_tags = pick_column(df, ["signal_hashtags", "article_hashtags", "llm_key_hashtags"])
col_discussion_tags = pick_column(df, ["discussion_hashtags"])
col_extracted = pick_column(df, ["article_text_extracted", "article_text_extracted?"])
col_stage = pick_column(df, ["signal_stage", "suggested_stage", "stage"])
col_domain = pick_column(df, ["source_domain"])
col_fetch_status = pick_column(df, ["fetch_status"])
col_tag_origin = pick_column(df, ["tag_origin"])
col_tag_review = pick_column(df, ["tag_review_status"])

if col_type is None:
    df["asset_type"] = "link"
    col_type = "asset_type"

if col_stage is None:
    df["signal_stage"] = "NA"
    col_stage = "signal_stage"

# Merge human review scores into the main dataset.
votes_df = vote_summary()
if col_id:
    df[col_id] = df[col_id].astype(str)
    df = df.merge(votes_df, left_on=col_id, right_on="signal_id", how="left", suffixes=("", "_votes"))
else:
    df["upvotes"] = 0
    df["downvotes"] = 0
    df["notes"] = 0
    df["score"] = 0

for review_col in ["upvotes", "downvotes", "notes", "score"]:
    if review_col not in df.columns:
        df[review_col] = 0
    df[review_col] = df[review_col].fillna(0).astype(int)

# Merge hashtags added through the website into the main dataset.
user_tags_df = user_tag_summary()
if col_id:
    df = df.merge(user_tags_df, left_on=col_id, right_on="signal_id", how="left", suffixes=("", "_user_tags"))
else:
    df["user_added_hashtags"] = ""
if "user_added_hashtags" not in df.columns:
    df["user_added_hashtags"] = ""
df["user_added_hashtags"] = df["user_added_hashtags"].fillna("")

# Downvotes act as vetoes: any veto pushes a record below all non-vetoed records.
df["vetoed"] = df["downvotes"] > 0
df["opinion_rank"] = np.where(df["vetoed"], -1_000_000 - df["downvotes"], df["upvotes"])

search_cols = [col_header, col_summary, col_tags, col_discussion_tags, col_channel, col_domain, "user_added_hashtags"]
df["search_text"] = df.apply(lambda row: build_search_text(row, search_cols), axis=1)

df["parsed_hashtags"] = df.apply(build_combined_hashtags, axis=1)
df["has_no_hashtag"] = df["parsed_hashtags"].apply(lambda tags: len(tags) == 0)

all_tags = sorted(set(tag for tags in df["parsed_hashtags"] for tag in tags), key=str.lower)

df["message_dt"] = pd.to_datetime(df[col_time], errors="coerce") if col_time else pd.NaT

with st.spinner("Preparing semantic search and clusters..."):
    embeddings = compute_embeddings(df["search_text"].fillna("").tolist())
    df = add_cluster_labels(df, embeddings, n_clusters=8)
    df = label_clusters(df, col_tags, col_header)

st.title("Signals")
explore_tab, overview_tab = st.tabs(["Explore", "Overview"])

with explore_tab:
    with st.sidebar:
        st.header("Filters")

        asset_types = sorted(df[col_type].dropna().astype(str).unique().tolist())
        selected_types = st.multiselect("Asset type", asset_types, default=asset_types)

        if col_channel:
            channels = sorted(
                df[col_channel].dropna().astype(str).unique().tolist(),
                key=lambda value: display_channel_label(value),
            )
            selected_channels = st.multiselect(
                "Channel",
                channels,
                default=channels,
                format_func=display_channel_label,
            )
        else:
            selected_channels = None

        if df["message_dt"].notna().any():
            min_date = df["message_dt"].min().date()
            max_date = df["message_dt"].max().date()
            selected_dates = st.date_input("Signal date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            selected_dates = None

        hashtag_options = ["No hashtag (N/A)"] + all_tags
        selected_hashtags = st.multiselect("Hashtag", hashtag_options, default=[])

        keyword_search = st.text_input("Keyword search", "")
        semantic_query = st.text_input("Semantic search", "")

        sort_by = st.selectbox(
            "Sort results by",
            [
                "Best by opinion",
                "Newest first",
                "Semantic relevance",
                "Most upvoted",
            ],
        )

    filtered = df.copy()
    filtered = filtered[filtered[col_type].astype(str).isin(selected_types)]

    if col_channel and selected_channels is not None:
        filtered = filtered[filtered[col_channel].astype(str).isin(selected_channels)]

    if selected_dates and df["message_dt"].notna().any():
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = end_date = selected_dates
        filtered = filtered[
            (filtered["message_dt"].dt.date >= start_date)
            & (filtered["message_dt"].dt.date <= end_date)
        ]

    if selected_hashtags:
        selected_real_tags = {tag.lower() for tag in selected_hashtags if tag != "No hashtag (N/A)"}
        include_no_tag = "No hashtag (N/A)" in selected_hashtags

        def hashtag_filter(tags):
            tag_set = {tag.lower() for tag in tags}
            return bool(tag_set & selected_real_tags) or (include_no_tag and not tag_set)

        filtered = filtered[filtered["parsed_hashtags"].apply(hashtag_filter)]

    if keyword_search:
        filtered = filtered[
            filtered["search_text"].astype(str).str.contains(keyword_search, case=False, na=False)
        ]

    if semantic_query.strip():
        query_embedding = compute_embeddings([semantic_query])[0]
        sims = cosine_similarity([query_embedding], embeddings)[0]
        df["semantic_score"] = sims
    else:
        df["semantic_score"] = np.nan

    if sort_by == "Semantic relevance" and semantic_query.strip():
        # Semantic relevance is still available when a semantic query is entered,
        # but vetoed items remain at the bottom within the semantic results.
        filtered = filtered.loc[df.loc[filtered.index, "semantic_score"].sort_values(ascending=False).index]
        if "vetoed" in filtered.columns:
            filtered = filtered.sort_values(["vetoed"], ascending=[True], kind="stable")
    elif sort_by in ["Best by opinion", "Most upvoted"]:
        # Upvotes push records up; any thumbs-down is treated as a veto and pushed below non-vetoed records.
        sort_cols = ["vetoed", "upvotes"]
        ascending = [True, False]
        if "message_dt" in filtered.columns:
            sort_cols.append("message_dt")
            ascending.append(False)
        filtered = filtered.sort_values(sort_cols, ascending=ascending, na_position="last")
    elif col_time and "message_dt" in filtered.columns:
        filtered = filtered.sort_values("message_dt", ascending=False, na_position="last")

    total_matching = len(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matching records", total_matching)
    c2.metric("Total records", len(df))
    c3.metric("Total unique hashtags", len(all_tags))
    c4.metric("Positive votes", int(df["upvotes"].sum()))

    # Pagination: the result set is no longer capped with .head().
    # Instead, all matching records are split into pages.
    page_size = st.selectbox("Records per page", [9, 18, 27, 36, 54], index=1)
    total_pages = max(1, int(np.ceil(total_matching / page_size)))

    if "results_page" not in st.session_state:
        st.session_state["results_page"] = 1
    if st.session_state["results_page"] > total_pages:
        st.session_state["results_page"] = total_pages
    if st.session_state["results_page"] < 1:
        st.session_state["results_page"] = 1

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="results_page",
    )

    start_idx = (int(page) - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered.iloc[start_idx:end_idx]

    st.caption(
        f"Showing records {start_idx + 1 if total_matching else 0}–{min(end_idx, total_matching)} "
        f"of {total_matching} across {total_pages} page(s)."
    )

    st.markdown("### Top hashtag pairs in matching records")
    st.caption(
        "Counts how many matching records contain both hashtags together. Each record counts once per pair."
    )
    pair_counts = hashtag_pair_frame(filtered, top_n=5)
    if pair_counts.empty:
        st.write("Not enough co-occurring hashtags in the current matching records.")
    else:
        st.dataframe(pair_counts, use_container_width=True, hide_index=True)

    st.markdown("## Results")
    if page_df.empty:
        st.info("No records match the current filters.")
    else:
        rows = list(page_df.iterrows())
        for start in range(0, len(rows), 2):
            cols = st.columns(2)
            for offset, (idx, row) in enumerate(rows[start:start + 2]):
                with cols[offset]:
                    render_signal_card(row, idx, semantic_query=semantic_query)
with overview_tab:
    st.markdown("## Overview")

    if col_time and df["message_dt"].notna().any():
        latest_time = df["message_dt"].max()
        recent_cutoff = latest_time - pd.Timedelta(days=30)
        recent = df[df["message_dt"] >= recent_cutoff].copy()
    else:
        latest_time = None
        recent = df.copy()

    timeframe = st.radio(
        "Overview timeframe",
        ["All time", "Last 30 days"],
        horizontal=True,
    )
    overview_df = df.copy() if timeframe == "All time" else recent.copy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signals shown", len(overview_df))
    m2.metric("All records", len(df))
    m3.metric("No hashtag (N/A)", int(overview_df["has_no_hashtag"].sum()) if "has_no_hashtag" in overview_df.columns else 0)
    m4.metric("Positive votes", int(overview_df["upvotes"].sum()))

    if latest_time is not None and timeframe == "Last 30 days":
        st.caption(f"Last-30-days window is anchored to latest record in dataset: {latest_time}")

    if col_channel and not overview_df.empty:
        st.markdown("### Signals by channel")
        channel_overview = overview_df.copy()
        channel_overview["display_channel"] = channel_overview[col_channel].apply(display_channel_label)
        channel_counts = count_frame(channel_overview, "display_channel", "channel", top_n=20, include_na=False)
        if not channel_counts.empty:
            render_sorted_bar_chart(channel_counts, "channel")
        else:
            st.write("No channel data available.")

    st.markdown("### Most common hashtags")
    tag_counter = Counter(tag for tags in overview_df["parsed_hashtags"] for tag in tags)
    if tag_counter:
        top_tags_df = pd.DataFrame(tag_counter.most_common(20), columns=["tag", "count"])
        render_sorted_bar_chart(top_tags_df, "tag")
    else:
        st.write("No hashtags available yet.")

    if col_domain:
        st.markdown("### Top source domains")
        domain_df = count_frame(overview_df, col_domain, "source_domain", top_n=20, include_na=False)
        if not domain_df.empty:
            render_sorted_bar_chart(domain_df, "source_domain")
        else:
            st.write("No source domains available.")

    if "upvotes" in overview_df.columns and overview_df["upvotes"].sum() > 0:
        st.markdown("### Highest-rated signals by your opinion")
        display_cols = []
        for candidate in [col_time, col_channel, col_header, col_domain, "upvotes", "notes", "vetoed"]:
            if candidate and candidate in overview_df.columns and candidate not in display_cols:
                display_cols.append(candidate)
        top_reviewed = overview_df.sort_values(["vetoed", "upvotes", "message_dt"], ascending=[True, False, False], na_position="last").head(10)
        st.dataframe(top_reviewed[display_cols], use_container_width=True, hide_index=True)

    if col_fetch_status:
        st.markdown("### Fetch status mix")
        fetch_df = count_frame(overview_df, col_fetch_status, "fetch_status", top_n=20, include_na=True)
        st.dataframe(fetch_df, use_container_width=True, hide_index=True)
