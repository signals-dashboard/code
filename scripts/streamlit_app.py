import hashlib
import os
import re
import streamlit as st
import pandas as pd
from supabase import create_client
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="CSF Signals Search", layout="wide")

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )

APP_DIR = Path(__file__).resolve().parent
# Works whether this file sits at the repo root or inside an app/ folder.
REPO_ROOT = APP_DIR if (APP_DIR / "data").exists() else APP_DIR.parent
CSV_PATH = REPO_ROOT / "data/processed/processed_signals.csv"
VOTES_PATH = REPO_ROOT / "data/processed/signal_votes.csv"
IMAGE_BASE_URL = os.getenv("CSF_IMAGE_BASE_URL", "").rstrip("/")


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
    """Extract tags from both hashtag strings and plain comma/semicolon tag lists.

    Older datasets may store AI/Ollama tags as "#AI #jobs" while newer or
    intermediate outputs may store them as "AI, jobs" or "AI | jobs". This
    parser normalises both forms so the UI can show Ollama tags reliably.
    """
    if text is None or pd.isna(text):
        return []

    raw = str(text).strip()
    if not raw or raw == "NA":
        return []

    hashtag_matches = re.findall(r"#[A-Za-z0-9_\-/]+", raw)
    if hashtag_matches:
        tags = hashtag_matches
    else:
        # Split plain tag lists while avoiding accidental extraction from prose.
        pieces = re.split(r"[,;|\n]+", raw)
        tags = []
        for piece in pieces:
            piece = piece.strip().strip("[](){}'\"")
            if not piece or piece.upper() == "NA":
                continue
            # Keep short tag-like phrases; drop long sentences.
            if len(piece.split()) > 4 or len(piece) > 40:
                continue
            piece = re.sub(r"\s+", "_", piece)
            piece = re.sub(r"[^A-Za-z0-9_\-/]", "", piece)
            if piece:
                tags.append("#" + piece.lstrip("#"))

    return [t.lower() if lower else t for t in tags]


def extract_tag_tokens(text, lower=True):
    """Backward-compatible alias used by the tag rendering logic."""
    return extract_hashtags(text, lower=lower)


def looks_like_metadata_fallback(text):
    """Detect fallback text that is metadata, not an article summary."""
    if not text or text == "NA":
        return True
    t = str(text).strip().lower()
    metadata_markers = [
        "available metadata:",
        "shared from ",
        "discussion context:",
    ]
    if any(marker in t for marker in metadata_markers):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    return False


def clean_summary_text(val, max_chars=260):
    """Trim display summary and suppress WhatsApp / metadata boilerplate."""
    text = short_text(val, max_chars=max_chars)
    if text == "NA":
        return "NA"

    text = re.sub(r"\s*Discussion context:\s*[^.。!?]*(?:[.。!?]|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Available metadata:\s*.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Shared from [^.]+\.\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    if looks_like_metadata_fallback(text):
        return "NA"
    return text or "NA"


def get_display_summary(row, max_chars=260):
    """Return only a real article summary for the card body.

    Rows with summary_source such as metadata_fallback, discussion_fallback, or
    none should not show the 'Shared from... Available metadata...' placeholder.
    This keeps the card body reserved for article summaries.
    """
    if not col_summary:
        return "NA"

    source = safe_text(row.get(col_summary_source)).lower() if col_summary_source else "unknown"
    if source and source not in ["article_text", "llm_summary", "ai_summary", "unknown", "na"]:
        return "NA"

    summary = clean_summary_text(row.get(col_summary), max_chars=max_chars)
    if summary == "NA" or looks_like_metadata_fallback(summary):
        return "NA"
    return summary

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


def unique_tags(tags):
    out = []
    seen = set()
    for tag in tags or []:
        if not tag:
            continue
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def get_tag_groups(row, max_tags=8):
    """Classify tags by tag_origin first, then read signal_hashtags.

    In this CSV, Ollama tags are stored in signal_hashtags. The source of those
    tags is not the column name, but tag_origin. So the order must be:

    1. read tag_origin
    2. read signal_hashtags
    3. colour/label the chips according to tag_origin
    """
    tag_origin = safe_text(row.get(col_tag_origin)).strip().lower() if col_tag_origin else "na"

    discussion_tags = extract_tag_tokens(row.get(col_discussion_tags), lower=False) if col_discussion_tags else []
    signal_tags = extract_tag_tokens(row.get(col_tags), lower=False) if col_tags else []
    explicit_ai_tags = extract_tag_tokens(row.get(col_ai_tags), lower=False) if col_ai_tags else []

    groups = []

    if tag_origin == "discussion_explicit" or tag_origin.startswith("discussion_explicit"):
        tags = discussion_tags or signal_tags
        tags = unique_tags(tags)[:max_tags]
        if tags:
            groups.append(("human", "Human tag", tags))

    elif tag_origin == "ai_generated_ollama" or "ollama" in tag_origin or tag_origin.startswith("ai_generated"):
        # This is the important branch: signal_hashtags becomes the blue Ollama layer.
        ai_tags = unique_tags(explicit_ai_tags or signal_tags)[:max_tags]
        if ai_tags:
            groups.append(("ai", "Ollama tag", ai_tags))

        # If the row also carries explicit discussion tags, show them separately.
        human_tags = [t for t in unique_tags(discussion_tags) if t.lower() not in {a.lower() for a in ai_tags}][:max_tags]
        if human_tags:
            groups.insert(0, ("human", "Human tag", human_tags))

    elif tag_origin.startswith("taxonomy_match"):
        tags = unique_tags(signal_tags)[:max_tags]
        if tags:
            groups.append(("taxonomy", "Matched tag", tags))

        human_tags = [t for t in unique_tags(discussion_tags) if t.lower() not in {a.lower() for a in tags}][:max_tags]
        if human_tags:
            groups.insert(0, ("human", "Human tag", human_tags))

    else:
        # Fallback for rows without a reliable tag_origin.
        human_tags = unique_tags(discussion_tags)[:max_tags]
        ai_tags = [t for t in unique_tags(explicit_ai_tags) if t.lower() not in {h.lower() for h in human_tags}][:max_tags]
        other_tags = [t for t in unique_tags(signal_tags) if t.lower() not in {h.lower() for h in human_tags} | {a.lower() for a in ai_tags}][:max_tags]

        if human_tags:
            groups.append(("human", "Human tag", human_tags))
        if ai_tags:
            groups.append(("ai", "Ollama tag", ai_tags))
        if other_tags:
            groups.append(("other", "Tag", other_tags))

    return groups

def render_chip_group(source, label, tags):
    styles = {
        "human": {
            "bg": "#dcfce7", "border": "#86efac", "text": "#166534", "label": "#15803d",
        },
        "ai": {
            "bg": "#e0f2fe", "border": "#7dd3fc", "text": "#075985", "label": "#0369a1",
        },
        "taxonomy": {
            "bg": "#f3f4f6", "border": "#d1d5db", "text": "#374151", "label": "#6b7280",
        },
        "other": {
            "bg": "#f5f3ff", "border": "#c4b5fd", "text": "#5b21b6", "label": "#6d28d9",
        },
    }
    stl = styles.get(source, styles["other"])
    chips = "".join(
        f'<span style="display:inline-block; margin:0 6px 6px 0; padding:4px 8px; '
        f'border-radius:999px; border:1px solid {stl["border"]}; background:{stl["bg"]}; '
        f'color:{stl["text"]}; font-size:0.78rem; font-weight:600;">{tag}</span>'
        for tag in tags
    )
    st.markdown(
        f'<div style="margin-top:0.25rem; margin-bottom:0.15rem;">'
        f'<span style="font-size:0.72rem; color:{stl["label"]}; font-weight:700; '
        f'margin-right:6px;">{label}</span>{chips}</div>',
        unsafe_allow_html=True,
    )


def render_tag_chips(row, max_tags=8):
    groups = get_tag_groups(row, max_tags=max_tags)
    for source, label, tags in groups:
        render_chip_group(source, label, tags)


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

        meta = []
        if col_domain:
            domain = safe_text(row.get(col_domain))
            if domain != "NA":
                meta.append(domain)
        if col_channel:
            channel = safe_text(row.get(col_channel))
            if channel != "NA":
                meta.append(channel)
        if col_time:
            time_val = safe_text(row.get(col_time))
            if time_val != "NA":
                meta.append(time_val)
        if meta:
            st.caption(" · ".join(meta))

        summary = get_display_summary(row, 260)
        if summary != "NA":
            st.markdown(summary)
        # Deliberately do not show metadata/discussion fallback text here.
        # The card body is reserved for article summaries.

        render_tag_chips(row, max_tags=6)

        signal_id = safe_text(row.get(col_id)) if col_id else str(idx)
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
                    save_vote(signal_id, "note", comment)
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
# Human review / voting helpers
# -----------------------------
VOTE_COLUMNS = ["signal_id", "vote", "comment", "timestamp"]


def widget_key(prefix, value):
    """Make short, stable Streamlit widget keys from long IDs/URLs."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_votes():
    if not VOTES_PATH.exists():
        return pd.DataFrame(columns=VOTE_COLUMNS)

    try:
        votes = pd.read_csv(VOTES_PATH)
    except Exception:
        return pd.DataFrame(columns=VOTE_COLUMNS)

    for col in VOTE_COLUMNS:
        if col not in votes.columns:
            votes[col] = ""
    return votes[VOTE_COLUMNS]


def save_vote(signal_id, vote, comment=""):
    VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    votes = load_votes()

    new_vote = pd.DataFrame([
        {
            "signal_id": str(signal_id),
            "vote": vote,
            "comment": str(comment or "").strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ])

    votes = pd.concat([votes, new_vote], ignore_index=True)
    votes.to_csv(VOTES_PATH, index=False)


def vote_summary():
    votes = load_votes()
    if votes.empty:
        return pd.DataFrame(columns=["signal_id", "upvotes", "downvotes", "notes", "score"])

    votes["signal_id"] = votes["signal_id"].astype(str)
    vote_counts = votes.pivot_table(
        index="signal_id",
        columns="vote",
        aggfunc="size",
        fill_value=0,
    ).reset_index()

    for col in ["up", "down", "note"]:
        if col not in vote_counts.columns:
            vote_counts[col] = 0

    vote_counts["upvotes"] = vote_counts["up"].astype(int)
    vote_counts["downvotes"] = vote_counts["down"].astype(int)
    vote_counts["notes"] = vote_counts["note"].astype(int)
    vote_counts["score"] = vote_counts["upvotes"] - vote_counts["downvotes"]

    return vote_counts[["signal_id", "upvotes", "downvotes", "notes", "score"]]


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


# -----------------------------
# Load data
# -----------------------------
df = load_data()
if df is None:
    st.error(f"Could not find {CSV_PATH}")
    st.stop()

# Flexible column matching
col_id = pick_column(df, ["signal_id", "record_id"])
col_time = pick_column(df, ["message_time", "time of message"])
col_type = pick_column(df, ["asset_type", "asset type"])
col_link = pick_column(df, ["final_url", "link_url", "link/image"])
col_image = pick_column(df, ["image_path", "image file", "local_image_path"])
col_header = pick_column(df, ["scraped_header", "scraped header of the link"])
col_channel = pick_column(df, ["sub_channel_name", "sub channel name"])
col_summary = pick_column(df, ["article_summary", "llm_summary", "llm summary of the article"])
col_summary_source = pick_column(df, ["summary_source"])
col_tags = pick_column(df, ["signal_hashtags", "article_hashtags", "llm_key_hashtags"])
col_discussion_tags = pick_column(df, ["discussion_hashtags"])
col_ai_tags = pick_column(df, [
    "ai_hashtags",
    "ollama_hashtags",
    "ollama_tags",
    "generated_hashtags",
    "generated_tags",
    "ai_tags",
    "llm_hashtags",
])
col_extracted = pick_column(df, ["article_text_extracted", "article_text_extracted?"])
col_stage = pick_column(df, ["signal_stage", "suggested_stage", "stage"])
col_domain = pick_column(df, ["source_domain"])
col_fetch_status = pick_column(df, ["fetch_status"])
col_tag_origin = pick_column(df, ["tag_origin", "tag-origin", "tag origin"])
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

# Downvotes act as vetoes: any veto pushes a record below all non-vetoed records.
df["vetoed"] = df["downvotes"] > 0
df["opinion_rank"] = np.where(df["vetoed"], -1_000_000 - df["downvotes"], df["upvotes"])

search_cols = [col_header, col_summary, col_tags, col_discussion_tags, col_ai_tags, col_channel, col_domain]
df["search_text"] = df.apply(lambda row: build_search_text(row, search_cols), axis=1)

def combined_row_hashtags(row):
    tags = []
    for c in [col_discussion_tags, col_ai_tags, col_tags]:
        if c and c in row.index:
            tags.extend(extract_hashtags(row.get(c)))
    return unique_tags(tags)

df["parsed_hashtags"] = df.apply(combined_row_hashtags, axis=1)

all_tags = sorted(set(tag for tags in df["parsed_hashtags"] for tag in tags))

df["message_dt"] = pd.to_datetime(df[col_time], errors="coerce") if col_time else pd.NaT

with st.spinner("Preparing semantic search and clusters..."):
    embeddings = compute_embeddings(df["search_text"].fillna("").tolist())
    df = add_cluster_labels(df, embeddings, n_clusters=8)
    df = label_clusters(df, col_tags, col_header)

st.title("CSF Signals Search")
explore_tab, overview_tab = st.tabs(["Explore", "Overview"])

with explore_tab:
    with st.sidebar:
        st.header("Filters")

        asset_types = sorted(df[col_type].dropna().astype(str).unique().tolist())
        selected_types = st.multiselect("Asset type", asset_types, default=asset_types)

        if col_channel:
            channels = sorted(df[col_channel].dropna().astype(str).unique().tolist())
            selected_channels = st.multiselect("Sub channel", channels, default=channels)
        else:
            selected_channels = None

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

    st.caption("Hashtags: green = original WhatsApp human tag; blue = Ollama/AI-generated tag; grey = taxonomy match.")

    # Pagination: the result set is no longer capped with .head().
    # Instead, all matching records are split into pages.
    page_size = st.selectbox("Records per page", [8, 16, 24, 32, 48], index=1)
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

    st.markdown("### Top hashtags on this page")
    current_tag_counter = Counter(tag for tags in page_df["parsed_hashtags"] for tag in tags)
    if current_tag_counter:
        top_tags_text = "  ".join([f"`{tag}` ({count})" for tag, count in current_tag_counter.most_common(15)])
        st.markdown(top_tags_text)
    else:
        st.write("No hashtags on this page.")

    st.markdown("### Clusters in matching records")
    cluster_counts = (
        filtered["cluster_label"]
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("All records", len(df))
    m2.metric("Last 30 days", len(recent))
    m3.metric("Needs tag review", int((df[col_tag_review] == "needs_review").sum()) if col_tag_review else 0)
    m4.metric("Positive votes", int(df["upvotes"].sum()))

    if latest_time is not None:
        st.caption(f"Recent window anchored to latest record in dataset: {latest_time}")

    if "upvotes" in df.columns and df["upvotes"].sum() > 0:
        st.markdown("### Highest-rated signals by your opinion")
        display_cols = []
        for candidate in [col_time, col_channel, col_header, col_domain, "upvotes", "notes", "vetoed"]:
            if candidate and candidate in df.columns and candidate not in display_cols:
                display_cols.append(candidate)
        top_reviewed = df.sort_values(["vetoed", "upvotes", "message_dt"], ascending=[True, False, False], na_position="last").head(10)
        st.dataframe(top_reviewed[display_cols], use_container_width=True, hide_index=True)

    if col_channel and not recent.empty:
        st.markdown("### Signals by sub-channel (last 30 days)")
        channel_counts = recent[col_channel].fillna("NA").astype(str).value_counts().rename_axis("sub_channel_name").reset_index(name="count")
        st.bar_chart(channel_counts.set_index("sub_channel_name"))

    if col_tags:
        st.markdown("### Most common signal hashtags")
        tag_counts = flatten_tags(recent[col_tags] if not recent.empty else df[col_tags])
        if tag_counts:
            top_tags_df = pd.DataFrame(tag_counts.most_common(15), columns=["tag", "count"]).set_index("tag")
            st.bar_chart(top_tags_df)
        else:
            st.write("No hashtags available yet.")

    if col_domain:
        st.markdown("### Top source domains")
        domain_df = (
            recent[col_domain].fillna("NA").astype(str)
            .replace("NA", pd.NA)
            .dropna()
            .value_counts()
            .head(15)
            .rename_axis("source_domain")
            .reset_index(name="count")
        )
        if not domain_df.empty:
            st.bar_chart(domain_df.set_index("source_domain"))
        else:
            st.write("No source domains available.")

    if col_fetch_status:
        st.markdown("### Fetch status mix")
        fetch_df = (
            df[col_fetch_status].fillna("NA").astype(str)
            .value_counts()
            .rename_axis("fetch_status")
            .reset_index(name="count")
        )
        st.dataframe(fetch_df, use_container_width=True, hide_index=True)
