import hashlib
import os
import re
import uuid
import google.generativeai as genai
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
from streamlit_echarts import st_echarts


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
            st.image(str(resolved), width='stretch')
            shown = True
            break

        if cand.lower().startswith(("http://", "https://")) and any(
            cand.lower().split("?")[0].endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]
        ):
            st.image(cand, width='stretch')
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

    # Forces the label to stay on one line and truncates with "..." if it gets too long
    truncated_label = f'<span style="display: inline-block; max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; vertical-align: bottom;">{label}</span>'

    st.markdown(chip_html(truncated_label, colour, bold=True), unsafe_allow_html=True)


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


def render_signal_card(row, idx, semantic_query="", key_prefix: str = "default", cluster_map: dict = None):
    """Render one signal as a compact height-controlled card for the grid layout."""
    signal_id = safe_text(row.get(col_id)) if col_id else str(idx)
    card_key_suffix = f"{key_prefix}_{signal_id}"

    with st.container(border=True):
        asset_type = safe_text(row.get(col_type))
        link = safe_text(row.get(col_link)) if col_link else "NA"

        # --- DOMAIN & DATE METADATA (Top of card for clean scanning) ---
        # Signal domain: currently read from sub_channel_name / channel.
        # Source domain below remains the website domain, e.g. ft.com or bloomberg.com.
        signal_domain = safe_text(row.get(col_channel)) if col_channel else "OTHERS"
        if signal_domain != "NA":
            render_domain_chip(signal_domain)
        
        # --- SOURCE & DATE (Strict 2-Row Layout) ---
        source_domain = safe_text(row.get(col_domain)) if col_domain else "Unknown Source"
        time_val = safe_text(row.get(col_time)) if col_time else "Unknown Date"
        
        # Parse the date and strip the time, converting to DD-MM-YYYY
        if time_val != "NA" and time_val != "Unknown Date":
            try:
                dt = pd.to_datetime(time_val)
                time_val = dt.strftime('%d/%m/%Y')
            except Exception:
                pass # If parsing fails, fall back to the original raw string
                
        # CSS Block: Forces exactly 2 lines total, with ellipsis on overflow for each line
        st.markdown(f"""
            <div style="
                line-height: 1.4;
                height: 2.8em; 
                font-size: 0.85rem;
                color: #555;
                margin-bottom: 0.5rem;
            ">
                <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {source_domain}
                </div>
                <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {time_val}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- TITLE (Truncated to enforce alignment) ---
        header = safe_text(row.get(col_header)) if col_header else "NA"
        is_image = (asset_type.lower() == "image")
        if header == "NA" and is_image:
            header = "Image signal"
            
        # CSS Block for a perfect 3-line fixed height
        title_html = f"""
            <div style="
                display: -webkit-box;
                -webkit-box-orient: vertical;
                -webkit-line-clamp: 3;
                overflow: hidden;
                line-height: 1.4;
                height: 4.2em; /* 3 lines * 1.4 line-height = Exactly 3 lines tall always */
                font-weight: 600;
                font-size: 1.05rem;
                margin-bottom: 0.5rem;
            ">{header}</div>
        """

        if is_image:    # shows the title
            # Split the row so the popover button sits next to the title
            title_col, btn_col = st.columns([3, 2])
            with title_col:
                st.markdown(title_html, unsafe_allow_html=True)
            with btn_col:
                with st.popover("View image", width='stretch'):
                    shown = show_image_from_candidates(get_image_candidates(row, col_link, col_image))
                    if not shown and is_image:
                        st.caption("Image not found in deployment")
        else:
            # Standard text signal
            st.markdown(title_html, unsafe_allow_html=True)

        # --- SUMMARY (Also height-controlled!) ---
        if col_summary:
            summary = safe_text(row.get(col_summary))
            if summary != "NA":
                # We apply the same CSS trick here but for 2 lines of smaller text
                st.markdown(f"""
                    <div style="
                        display: -webkit-box;
                        -webkit-box-orient: vertical;
                        -webkit-line-clamp: 2;
                        overflow: hidden;
                        line-height: 1.4;
                        height: 2.8em; 
                        font-size: 0.9rem;
                        color: #555;
                    ">{summary}</div>
                """, unsafe_allow_html=True)

        # --- HASHTAGS & POPOVER (Inline Row) ---
        parsed = build_combined_hashtags(row)

        # Create columns to put the "+" popover on the same line as the tags
        tag_col, btn_col = st.columns([5, 1])

        with tag_col:
            if parsed:
                # Initialize session state for this specific card's tags
                state_key = f"expand_tags_{idx}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = False
                
                # Logic: Show first 2 tags, provide a toggle button if there are more
                if len(parsed) > 2 and not st.session_state[state_key]:
                    tags, more = st.columns([3, 1])
                    with tags:
                        render_tag_bubbles(parsed[:2], signal_domain=signal_domain, domain_aware=True)
                    
                        # Note: If your Streamlit version is 1.37+, you can add `type="tertiary"` 
                        # to make this button look exactly like a borderless hyperlink!
                    with more:
                        if st.button(f"{len(parsed) - 2} more...", key=f"btn_more_{idx}", type="tertiary"):
                            st.session_state[state_key] = True
                            st.rerun()
                else:
                    render_tag_bubbles(parsed, signal_domain=signal_domain, domain_aware=True)
                    if len(parsed) > 2:
                        if st.button("Show less", key=f"btn_less_{idx}", type="tertiary"):
                            st.session_state[state_key] = False
                            st.rerun()            
            else:
                st.caption("No tags")
                # render_bubbles(["No tags"], max_items=1)

        with btn_col:
            # The popover creates a floating menu that DOES NOT break grid alignment!
            with st.popover("➕"):
                new_tags = st.text_input(
                    "Add hashtag",
                    key=f"add_tag_{card_key_suffix}",
                    placeholder="Add hashtags e.g. #AI, #Economy",
                    label_visibility="collapsed"
                )
                if st.button("Save", key=f"save_tag_{card_key_suffix}", width='stretch'):
                    saved = save_user_hashtags(signal_id, new_tags)
                    if saved:
                        st.rerun()
                    
        # 1. Fallback: If cluster_map wasn't passed as an argument, build it from global dataframe
        if 'cluster_map' not in locals() or not cluster_map:
            cluster_map = dict(zip(clusters_df['id'].astype(str), clusters_df['title']))
        
        select_col, attach_button_col = st.columns([5, 1])
        # 2. Searchable Selectbox (index=None leaves it blank by default!)
        with select_col:
            selected_target_cid = st.selectbox(
                "Attach to an Existing Cluster",
                options=list(cluster_map.keys()),
                format_func=lambda cid: cluster_map.get(cid, "Unknown Cluster"),
                index=None,  # Modern Streamlit: Starts empty so they must actively search/choose!
                placeholder="Attach to an Existing Cluster",
                key=f"select_cluster_{card_key_suffix}",
                label_visibility="collapsed"
            )
        
        # 3. Action Button
        with attach_button_col:
            if st.button("🎯", key=f"btn_bind_{card_key_suffix}", type="primary", use_container_width=True):
                if selected_target_cid:
                    # Call your modular function passing the single ID as a 1-item list!
                    success, msg = add_signals_to_cluster([signal_id], selected_target_cid)
                    
                    if success:
                        st.toast(msg)
                        # ---> CRITICAL: Clear cache so the graph & attached counts update instantly! <---
                        with st.spinner("🤖 Regenerating Evolving Synthesis..."):
                            success, msg = trigger_evolving_synthesis(selected_target_cid, signals_df)
                        if success:
                            st.toast(msg)
                        else:
                            st.error(msg)
                        load_cluster_graph_data.clear() 
                        st.rerun()
                    else:
                        st.toast(msg)
                else:
                    st.toast("⚠️ Please select a cluster from the dropdown first.")

        # Removed temporarily to streamline results view.
        # upvotes = int(row.get("upvotes", 0))
        # downvotes = int(row.get("downvotes", 0))
        # notes = int(row.get("notes", 0))
        # veto_label = " | Vetoed" if downvotes > 0 else ""

        # with st.expander(f"Your Opinion: 👍 {upvotes} | Notes {notes}{veto_label}", expanded=False):
        #     vote_col1, vote_col2 = st.columns(2)
        #     with vote_col1:
        #         if st.button("👍 Useful / emerging", key=widget_key("up", signal_id)):
        #             save_vote(signal_id, "up")
        #             st.success("Vote saved.")
        #             st.rerun()
        #     with vote_col2:
        #         if st.button("👎 Not useful", key=widget_key("down", signal_id)):
        #             save_vote(signal_id, "down")
        #             st.warning("Vote saved.")
        #             st.rerun()

        #     comment = st.text_input(
        #         "Optional note",
        #         key=widget_key("comment", signal_id),
        #         placeholder="Why is this useful, emerging, noisy, or irrelevant?",
        #     )
        #     if st.button("Save note", key=widget_key("note", signal_id)):
        #         if comment.strip():
        #             save_note(signal_id, comment)
        #             st.success("Note saved.")
        #             st.rerun()
        #         else:
        #             st.info("Write a note before saving.")

        # --- 6. FOOTER (Flexbox Baseline Alignment) ---
        score_text = ""
        if semantic_query.strip() and "semantic_score" in signals_df.columns:
            score = signals_df.loc[idx, "semantic_score"]
            if pd.notna(score):
                score_text = f"Score: {score:.2f}"
                
        link_html = ""
        if link != "NA":
            link_html = f"<a href='{link}' target='_blank' style='text-decoration: none; color: inherit;'>Open Source ↗</a>"

        # display: flex + justify-content: space-between forces them to opposite edges on the exact same baseline!
        st.markdown(f"""
            <div style="
                display: flex; 
                justify-content: space-between; 
                alrgn-items: baseline; 
                width: 100%; 
                font-size: 0.85rem; 
                color: rgba(128, 128, 128, 0.85); 
                padding-top: 0.3rem; 
                padding-bottom: 0.5rem; 
                border-top: 1px solid rgba(128, 128, 128, 0.15);
                margin-top: 0.1rem; 
                margin-bottom: 0.2rem;
            ">
                <div>{score_text}</div>
                <div>{link_html}</div>
            </div>
        """, unsafe_allow_html=True)

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
            "created_by": get_current_user()
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
            "created_by": get_current_user(),
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
            "created_by": get_current_user(),
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
    st.altair_chart(chart, width='stretch')


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

# Loads tables "clusters" and "cluster_signals" from Supabase
@st.cache_data(ttl=60, show_spinner=False)
def load_cluster_graph_data():
    """
    Fetches core clusters and junction links from Supabase.
    Safely enforces column schemas to prevent KeyError on empty database tables.
    """
    # Define your expected schemas so Pandas never creates column-less DataFrames
    CLUSTER_COLS = ["id", "title", "maturity_status", "evolving_synthesis", "created_by", "created_at"]
    LINK_COLS = ["id", "cluster_id", "signal_id", "added_by", "added_at"]
    INSIGHTS_COLS = ["id", "cluster_id", "content", "added_by", "created_at"]

    try:
        supabase = get_supabase()
        
        # 1. Fetch Core Clusters
        clusters_res = supabase.table("clusters").select("*").execute()
        if clusters_res.data:
            clusters_df = pd.DataFrame(clusters_res.data)
        else:
            clusters_df = pd.DataFrame(columns=CLUSTER_COLS)
        
        # 2. Fetch Junction Links (The Edges!)
        links_res = supabase.table("cluster_signals").select("*").execute()
        if links_res.data:
            cluster_signals_df = pd.DataFrame(links_res.data)
        else:
            cluster_signals_df = pd.DataFrame(columns=LINK_COLS)
        
        # 3. Fetch Cluster Insights
        insights_res = supabase.table("cluster_insights").select("*").execute()
        if insights_res.data:
            cluster_insights_df = pd.DataFrame(insights_res.data)
        else:
            cluster_insights_df = pd.DataFrame(columns=INSIGHTS_COLS)

        return clusters_df, cluster_signals_df, cluster_insights_df
    
        
    except Exception as exc:
        st.error(f"Failed to load cluster data from Supabase: {exc}")
        # Return empty DataFrames WITH columns instead of None to prevent cascading crashes
        return pd.DataFrame(columns=CLUSTER_COLS), pd.DataFrame(columns=LINK_COLS)

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
# Pagination functions
# -----------------------------
def get_pagination_window(current_page, total_pages):
    """Calculates which page numbers and ellipsis (...) to display."""
    # If 5 pages or fewer, just show all numbers
    if total_pages <= 5:
        return list(range(1, total_pages + 1))
    
    # If > 7 pages, use a +-3 window (7 pages wide). If 6-7 pages, use a +-2 window.
    window_size = 3 if total_pages > 7 else 2
    
    start = max(1, current_page - window_size)
    end = min(total_pages, current_page + window_size)
    
    # Edge case adjustment: keep the window width consistent when near Page 1 or the Last Page
    if start == 1:
        end = min(total_pages, 1 + (window_size * 2))
    elif end == total_pages:
        start = max(1, total_pages - (window_size * 2))
        
    pages = list(range(start, end + 1))
    
    # Prepend or append the ellipsis if there is a gap to the outer edges
    if start > 1:
        pages = ["..."] + pages
    if end < total_pages:
        pages = pages + ["..."]
        
    return pages

def render_pagination(total_pages, key_prefix="bottom"):
    """Renders the dynamic pagination bar with arrows and unclickable ellipsis."""
    if total_pages <= 1:
        return # Hide pagination entirely if there is only 1 page of results!
        
    nav_items = []
    
    # 1. Add Back/First arrows only if we are NOT on Page 1
    if st.session_state['current_page'] > 1:
        nav_items.extend(["<<", "<"])
        
    # 2. Add our smart sliding window of page numbers and "..."
    nav_items.extend(get_pagination_window(st.session_state['current_page'], total_pages))
        
    # 3. Add Next/Last arrows only if we are NOT on the final page
    if st.session_state['current_page'] < total_pages:
        nav_items.extend([">", ">>"])

    # 4. Create flexible columns to auto-center the bar
    spacer_left, *btn_cols, spacer_right = st.columns([1] + [0.4] * len(nav_items) + [1])
    
    # 5. Render the buttons or filler text inside their respective columns
    for col, item in zip(btn_cols, nav_items):
        with col:
            if item == "...":
                # Renders as unclickable, styled HTML filler text instead of a button!
                st.markdown(
                    "<div style='text-align: center; padding-top: 8px; font-weight: bold; color: #888;'>...</div>", 
                    unsafe_allow_html=True
                )
            elif isinstance(item, int):
                # Page Number Button
                btn_type = "primary" if item == st.session_state['current_page'] else "secondary"
                if st.button(str(item), key=f"{key_prefix}_page_{item}", type=btn_type, width='stretch'):
                    st.session_state['current_page'] = item
                    st.rerun()
            else:
                # Arrow Navigation Buttons (<<, <, >, >>)
                if st.button(item, key=f"{key_prefix}_nav_{item}", width='stretch'):
                    if item == "<<":
                        st.session_state['current_page'] = 1
                    elif item == "<":
                        st.session_state['current_page'] -= 1
                    elif item == ">":
                        st.session_state['current_page'] += 1
                    elif item == ">>":
                        st.session_state['current_page'] = total_pages
                    st.rerun()

# -----------------------------
# User login and storage helper function and variable
# -----------------------------

# HARDCODED ANALYST LIST (EDIT WHEN NEEDED)
ANALYSTS = sorted([
    "TERENCE", "ANGEL", "SEEMA", 
    "CHARLENE", "HAO GUANG",  "FUAD", 
    "XUE TING", "GURU", "JEVON", 
    "YUN HUI", "JAKIN", "RIQQAH", 
    "MATTHEW", "GWYNETH"
    ])

# Global helper function to get current_user
def get_current_user():
    return st.session_state.get("current_user", "Anonymous Analyst")

# -----------------------------
# Cluster helper functions
# -----------------------------
def save_new_cluster(title, status, signal_ids):
    """
    Saves a new cluster and batch-attaches any selected signals.
    Returns: (bool success, str message)
    """
    try:
        supabase = get_supabase()
        current_user = get_current_user()
        clean_title = title.strip()

        # 1. Duplicate Title Check
        # .ilike() performs a case-insensitive search (prevents "AI" vs "ai" duplicates)
        existing = supabase.table("clusters").select("id").ilike("title", clean_title).execute()
        
        if existing.data and len(existing.data) > 0:
            return False, f"⚠️ A cluster named '{clean_title}' already exists! Please choose a unique title."
        
        # 2. Generate a unique ID for the cluster right here in Python
        cluster_id = str(uuid.uuid4())
        
        # 3. Insert the core cluster row
        cluster_payload = {
            "id": cluster_id,
            "title": title.strip(),
            "maturity_status": status,
            "created_by": current_user,
            # 'created_at' is omitted so Supabase auto-stamps it with now()
        }
        supabase.table("clusters").insert(cluster_payload).execute()
        
        # 4. If signals were selected, build a batch array and insert in ONE network call
        if signal_ids:
            junction_records = [
                {
                    "id": str(uuid.uuid4()),
                    "cluster_id": cluster_id,
                    "signal_id": str(sig_id),
                    "added_by": current_user
                }
                for sig_id in signal_ids
            ]
            supabase.table("cluster_signals").insert(junction_records).execute()
            
        return True, f"Successfully created '{title}' with {len(signal_ids)} attached signals!"
        
    except Exception as exc:
        return False, f"Could not save cluster to Supabase: {exc}"

# Function for generating evolving synthesis
def trigger_evolving_synthesis(cluster_id, signals_df):
#     """
#     Gathers cluster title, attached signals, and human insights,
#     generates an updated synthesis via LLM, and updates Supabase.
#     """
#     try:
#         supabase = get_supabase()
        
#         # 1. Gather Context
#         c_res = supabase.table("clusters").select("*").eq("id", cluster_id).execute()
#         if not c_res.data: return
#         cluster = c_res.data[0]
        
#         # Get attached signals
#         links = supabase.table("cluster_signals").select("signal_id").eq("cluster_id", cluster_id).execute()
#         sig_ids = [str(item['signal_id']) for item in links.data]
#         # Use this when migrated signals database to supabase. Meanwhile use df
#         # signals = supabase.table("signals").select("search_text").in_("id", sig_ids).execute().data if sig_ids else []
#         signals = signals_df[signals_df['signal_id'].astype(str) in sig_ids]
        
#         # Get insights log
#         i_res = supabase.table("cluster_insights").select("*").eq("cluster_id", cluster_id).execute
#         if not i_res.data: return
#         insights = i_res.data

#         # TODO Configure the API key securely from Streamlit secrets
#         genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
#         model = genai.GenerativeModel('gemini-1.5-flash')
        
#         # 2. Build the LLM Memory Payload
#         sig_texts = "\n---\n".join([s.get('search_text', '') for s in signals])
#         insights_text = "\n".join([f"[{i.get('created_at')}] {i.get('added_by')}: {i.get('content')}" for i in insights])
        
#         prompt = f"""
#         You are a Strategic Horizon Scanning AI. Generate an 'Evolving Synthesis' (max 100 words) 
#         for the following strategic foresight cluster. Analyze the overarching theme, synthesize the 
#         evidence from attached signals, and incorporate expert human analyst insights provided.
        
#         CLUSTER TITLE: {cluster['title']}
        
#         ATTACHED SIGNALS EVIDENCE BASE:
#         {sig_texts if sig_texts else "No signals attached yet."}
        
#         ANALYST TRIAGE LOG & HUMAN INSIGHTS:
#         {insights_text if insights_text else "No human insights recorded yet."}
        
#         Write a cohesive, executive-ready analytical paragraph:
#         """
        
#         # 3. Call your LLM SDK (Example using generic chat completion)
#         response = model.generate_content(prompt)
#         generated_synthesis = response.text.strip()
        
#         # Mocking output for structure:
#         # generated_synthesis = f"Strategic synthesis for '{cluster['title']}' integrating {len(sig_ids)} signals and {len(insights)} analyst insights..."
        
#         # 4. Save back to Supabase
#         supabase.table("clusters").update({"evolving_synthesis": generated_synthesis}).eq("id", cluster_id).execute()
        
#         return True, "✅ Evolving synthesis regenerated!"
    
#     except Exception as exc:
#         return False, f"Synthesis generation failed: {exc}"
    
    # Just for testing
    return True, "Works fine"


def update_cluster_title(new_title: str, target_cid: str) -> tuple[bool, str]:
    """
    Updates the title of an existing cluster in the 'clusters' table.
    """
    try:
        supabase = get_supabase()
        clean_title = new_title.strip()
        
        if not clean_title:
            return False, "⚠️ Title cannot be empty."
            
        # Optional: Prevent renaming to a title that already belongs to ANOTHER cluster
        existing = supabase.table("clusters").select("id").ilike("title", clean_title).neq("id", target_cid).execute()
        if existing.data and len(existing.data) > 0:
            return False, f"⚠️ Another cluster named '{clean_title}' already exists."

        supabase.table("clusters").update({"title": clean_title}).eq("id", target_cid).execute()
        return True, "✅ Cluster title updated successfully!"
        
    except Exception as exc:
        return False, f"❌ Database error updating title: {exc}"


def update_cluster_status(new_status: str, target_cid: str) -> tuple[bool, str]:
    """
    Updates the maturity status of an existing cluster in the 'clusters' table.
    """
    try:
        supabase = get_supabase()
        supabase.table("clusters").update({"maturity_status": new_status}).eq("id", target_cid).execute()
        return True, f"✅ Maturity status moved to {new_status}!"
        
    except Exception as exc:
        return False, f"❌ Database error updating status: {exc}"


def add_cluster_insight(new_insight_text: str, target_cid: str) -> tuple[bool, str]:
    """
    Inserts a new analyst insight into the relational 'cluster_insights' table.
    Note: Requires target_cid to act as the foreign key linking the insight to the cluster.
    """
    try:
        supabase = get_supabase()
        clean_text = new_insight_text.strip()
        current_user = get_current_user()
        
        if not clean_text:
            return False, "⚠️ Insight content cannot be empty."

        payload = {
            "id": str(uuid.uuid4()),
            "cluster_id": target_cid,
            "added_by": current_user,
            "content": clean_text
        }
        
        supabase.table("cluster_insights").insert(payload).execute()
        return True, "✅ Insight added to log!"
        
    except Exception as exc:
        return False, f"❌ Database error saving insight: {exc}"


def add_signals_to_cluster(selected_to_attach: list, target_cid: str) -> tuple[bool, str]:
    """
    Batch inserts new signal-to-cluster mappings into the 'cluster_signals' junction table.
    """
    if not selected_to_attach:
        return False, "⚠️ No signals selected to attach."
        
    try:
        supabase = get_supabase()
        current_user = get_current_user()

        # 1. Fetch existing links to prevent SQL primary/composite key uniqueness crashes
        existing_res = supabase.table("cluster_signals").select("signal_id").eq("cluster_id", target_cid).execute()
        existing_sig_ids = {str(row['signal_id']) for row in existing_res.data} if existing_res.data else set()
        
        # 2. Filter out signals that are already attached to this cluster
        new_signals = [str(sid) for sid in selected_to_attach if str(sid) not in existing_sig_ids]
        
        if not new_signals:
            return False, "ℹ️ Selected signals are already attached to this cluster."

        # 3. Build batch insert payload
        junction_records = [
            {
                "id": str(uuid.uuid4()),
                "cluster_id": target_cid,
                "signal_id": sid,
                "added_by": current_user
            }
            for sid in new_signals
        ]
        
        supabase.table("cluster_signals").insert(junction_records).execute()
        return True, f"✅ Successfully linked {len(new_signals)} new signal(s)!"
        
    except Exception as exc:
        return False, f"❌ Database error linking signals: {exc}"

# Function for building node visualisation
@st.cache_data(show_spinner=False)
def build_cluster_topology(clusters_df, cluster_signals_df, signals_df, selected_cluster_ids, isolate_mode):
    nodes = []
    links = []
    added_node_ids = set()

    # Track which signals belong to selected clusters so we can show their labels by default!
    highlighted_signal_ids = set()

    # --- STEP 1: FILTER CLUSTERS BASED ON UI SELECTION ---
    if isolate_mode and selected_cluster_ids:
        # ISOLATE: Only show the clusters explicitly chosen in the multiselect
        active_clusters = clusters_df[clusters_df['id'].astype(str).isin(selected_cluster_ids)]
    else:
        # NORMAL: Show all clusters (or filter by your maturity multiselect here!)
        active_clusters = clusters_df

    # --- STEP 2: BUILD CLUSTER NODES (THE HUBS) ---
    for _, row in active_clusters.iterrows():
        cid = str(row['id'])
        is_selected = cid in selected_cluster_ids if selected_cluster_ids else False

        # Determine Visual Highlighting
        if is_selected:
            node_color = "#9B51E0"  # Vibrant Purple for selected/searched clusters
            size = 55               # Make them physically pop
            # Place selected nodes at (0,0) so the physics engine centers them!
            coords = {"x": 0, "y": 0, "fixed": False} 
            shadow = {"shadowBlur": 20, "shadowColor": "rgba(155, 81, 224, 0.5)"}
        elif selected_cluster_ids:
            node_color = "#FF8A8A"  # Muted pink/rose for unselected clusters when a search is active
            size = 40
            coords = {}
            shadow = {}
        else:
            node_color = "#FF4B4B"  # Default Streamlit Red when nothing is searched
            size = 45
            coords = {}
            shadow = {}

        node_dict = {
            "id": cid,
            "name": row['title'],
            "node_type": "cluster",
            "symbolSize": size,
            "itemStyle": { "color": node_color, **shadow },
            # REQUIREMENT 3a: Cluster titles always shown and BOLDED
            "label": { 
                "show": True, 
                "position": "right", 
                "fontWeight": "bold",
                "fontSize": 13
            }, 
        }
        node_dict.update(coords)
        nodes.append(node_dict)
        added_node_ids.add(cid)

    # --- STEP 3: BUILD EDGE LINKS & SIGNAL NODES (THE LEAVES) ---
    # Identify signals connected to selected clusters
    if selected_cluster_ids:
        selected_links = cluster_signals_df[cluster_signals_df['cluster_id'].astype(str).isin(selected_cluster_ids)]
        highlighted_signal_ids = set(selected_links['signal_id'].astype(str))

    # Find all junction rows connected to our currently active clusters
    active_links = cluster_signals_df[cluster_signals_df['cluster_id'].astype(str).isin(added_node_ids)]

    for _, row in active_links.iterrows():
        cid = str(row['cluster_id'])
        sid = str(row['signal_id'])

        # 1. Create the Edge Link
        links.append({
            "source": cid,
            "target": sid,
            "lineStyle": { "width": 1.5, "curveness": 0.1 } # Slight curve looks elegant
        })

        # 2. If we haven't added this Signal Node to the graph yet, add it now!
        if sid not in added_node_ids:
            # Look up the signal's title from your master repository DataFrame
            sig_match = signals_df[signals_df['signal_id'].astype(str) == sid]
            sig_title = sig_match.iloc[0]['search_text'] if not sig_match.empty else f"Signal #{sid[:6]}"
            
            # Show label by default ONLY if connected to a searched cluster!
            show_signal_label = sid in highlighted_signal_ids

            nodes.append({
                "id": sid,
                "name": sig_title[:45] + "...", # Truncate long signal titles on the canvas
                "node_type": "signal",
                "symbolSize": 18, # Smaller dots for signals
                "itemStyle": { "color": "#1C83E1" }, # Cool blue for signals
                "label": {
                    "show": show_signal_label,
                    "position": "right", 
                    "fontSize": 11
                },
                "emphasis": {
                    "label": { "show": True, "fontWeight": "bold" }
                }
            })
            added_node_ids.add(sid)

    return nodes, links

# Renders the whole cluster view when a cluster node or card is clicked
def render_cluster_dashboard(cid, clusters_df, cluster_signals_df, signals_df, key_prefix: str = "primary"):
    target_cid = cid
    matched_clusters = clusters_df[clusters_df['id'].astype(str).str.strip().str.lower() == target_cid]
    
    if matched_clusters.empty:
        st.error(f"❌ Error: Node ID `{target_cid}` was clicked, but no matching UUID exists in `clusters_df`!")
    else:
        st.markdown(f"### 🔎 Inspecting Cluster: `#{target_cid[:6]}`")

        cluster_row = matched_clusters.iloc[0]
        
        # Grab attached signals for this cluster
        attached_links = cluster_signals_df[cluster_signals_df['cluster_id'].astype(str) == target_cid]
        attached_sig_ids = attached_links['signal_id'].astype(str).tolist()
        attached_signals_df = signals_df[signals_df['signal_id'].astype(str).isin(attached_sig_ids)]

        col_details, col_insights = st.columns([1.1, 0.9])
        
        # =========================================================
        # COLUMN 1: CLUSTER DETAILS & MICRO-EVIDENCE
        # =========================================================
        with col_details:
            # 1. Editable Title Block
            with st.container(border=True):
                title_header_col, edit_btn_col = st.columns([5, 1])
                with title_header_col:
                    st.markdown(f"## {cluster_row['title']}")
                with edit_btn_col:
                    with st.popover("✏️ Edit"):
                        st.markdown("#### Rename Cluster")
                        new_title_input = st.text_input(
                            "Title", 
                            value=cluster_row['title'], 
                            label_visibility="collapsed",
                            key=f"input_title_{key_prefix}_{target_cid}"
                            )
                        if st.button("💾 Save", type="primary", use_container_width=True):
                            if new_title_input.strip() and new_title_input != cluster_row['title']:
                                # 1. Update Supabase
                                success, msg = update_cluster_title(new_title_input.strip(), target_cid)
                                if success:
                                    st.toast(msg)
                                    # 2. Trigger AI Synthesis (Feature 3)
                                    with st.spinner("🤖 Regenerating Evolving Synthesis..."):
                                        success, msg = trigger_evolving_synthesis(current_id, signals_df)
                                    if success:
                                        st.toast(msg)
                                    else:
                                        st.error(msg)
                                    load_cluster_graph_data.clear()
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                # 2. Metadata Metrics
                m1, m2 = st.columns(2)
                raw_date = str(cluster_row.get('created_at', 'Unknown'))[:10]
                try:
                    # Convert YYYY-MM-DD to DD/MM/YYYY
                    formatted_date = datetime.strptime(raw_date, "%Y-%m-%d").strftime("%d/%m/%Y")
                except ValueError:
                    formatted_date = raw_date
                m1.metric("Created", formatted_date)
                m2.metric("Evidence Base", f"{len(attached_sig_ids)} Signals")
                
                # 3. Editable Status Dropdown
                status_options = ["Active", "KIV", "Presentation-ready", "Archived"]
                curr_status = cluster_row.get('maturity_status', 'Active')
                curr_status_idx = status_options.index(curr_status) if curr_status in status_options else 0
                
                new_status = st.selectbox(
                    "Maturity Status", 
                    status_options, 
                    index=curr_status_idx, 
                    key=f"status_{key_prefix}_{target_cid}")
                if new_status != curr_status:
                    # Run Supabase update here!
                    success, msg = update_cluster_status(new_status, target_cid)
                    if success:
                        st.toast(msg)
                        load_cluster_graph_data.clear()
                        st.rerun()
                    else:
                        st.error(msg)

                # 4. Evolving Synthesis
                st.markdown("#### Evolving Synthesis")
                synthesis_text = cluster_row.get('evolving_synthesis', 'No automated synthesis generated yet. Attach more signals to trigger AI pattern detection.')
                st.info(synthesis_text)
                
                # 5. Micro-Evidence List (Condensed Signal View)
                st.markdown("#### Attached Signal Summary")
                with st.container(height=220, border=True):
                    if attached_signals_df.empty:
                        st.caption("No signals attached yet.")
                    else:
                        for _, sig in attached_signals_df.iterrows():
                            # Map your category hex colors cleanly
                            raw_domain = str(sig.get(col_channel, 'OTHERS')).strip().upper()
                            
                            clean_domain = normalise_domain(raw_domain)
                            dot_color = DOMAIN_COLOURS.get(clean_domain, DOMAIN_COLOURS["OTHERS"])
                            
                            cat_name = DOMAIN_LABELS.get(clean_domain, 'OTHERS')
                            
                            short_title = str(sig.get('search_text', 'Untitled'))[:50] + "..."
                            
                            st.markdown(
                                f"""<div style='line-height: 1.4; margin-bottom: 8px; font-size: 0.85rem;'>
                                <span style='color: {dot_color}; font-size: 1.1rem;'>●</span> 
                                <b>[{cat_name}]</b> {short_title}
                                </div>""", 
                                unsafe_allow_html=True
                            )

        # =========================================================
        # COLUMN 2: HUMAN INSIGHTS LOG
        # =========================================================
        with col_insights:
            with st.container(border=True):
                st.markdown("#### Analyst Insight Log")
                
                # 1. Scrollable Log Container
                with st.container(height=420, border=True):
                    # Assume you loaded an `insights_list` (df of rows) from Supabase for this cluster
                    insights_rows = cluster_insights_df[cluster_insights_df["cluster_id"].astype(str) == target_cid]
                    
                    if insights_rows.empty:
                        st.caption("No human insights recorded yet. Be the first to add strategic context below!")
                    else:
                        for _, log in insights_rows.iterrows():
                            with st.container(border=True):
                                author = log.get('added_by', 'Anonymous Analyst')
                                timestamp = datetime.strptime(log.get('created_at', 'Unknown')[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
                                st.markdown(
                                    f"""
                                    <div style='line-height: 1.3; margin-bottom: 2px;'>
                                        <small style='color: #808495; font-size: 0.75rem;'><b>{author}</b> • {timestamp}</small>
                                    </div>
                                    <div style='font-size: 0.88rem; line-height: 1.4; margin-bottom: 16px; color: var(--text-color);'>
                                        {log.get('content', '')}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )

                # 2. Add New Insight Input Block
                with st.form(key=f"add_insight_form_{key_prefix}_{target_cid}", clear_on_submit=True):
                    new_insight_text = st.text_area("Add Strategic Insight", placeholder="Note potential connections, implications, blind spots, or verification needs etc...", height=80, label_visibility="collapsed")
                    submit_insight = st.form_submit_button("Add Insight", use_container_width=True, type="primary")
                    
                    if submit_insight and new_insight_text.strip():                        
                        # 1. Update Supabase
                        success, msg = add_cluster_insight(new_insight_text.strip(), target_cid)
                        if success:
                            st.toast(msg)
                            # 2. Trigger AI Synthesis
                            with st.spinner("🤖 Regenerating Evolving Synthesis..."):
                                success, msg = trigger_evolving_synthesis(current_id, signals_df)
                            if success:
                                st.toast(msg)
                            else:
                                st.error(msg)
                            load_cluster_graph_data.clear()
                            st.rerun()
                        else:
                            st.error(msg)    

        # =========================================================
        # FULL WIDTH BELOW: SIGNAL MANAGEMENT EXPANDER
        # =========================================================
        st.markdown("")
        with st.expander(f"Manage Cluster Signals ({len(attached_sig_ids)})", expanded=True):
            
            st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)    # same formatting style as the signal repository search expander
            
            # --- TOP: SEARCH & ADD SIGNAL FUNCTIONALITY ---
            st.markdown("##### Attach Additional Existing Signals")

            # 1a. Provide a quick keyword filter so analysts don't have to scroll through hundreds of rows
            if 'manage_form_id' not in st.session_state:
                st.session_state['manage_form_id'] = 0

            manage_form_id = st.session_state['manage_form_id']

            manage_search = st.text_input(
                "Filter signals by keyword and select them below", 
                key=f"manage_signal_search_{key_prefix}_{manage_form_id}", 
                placeholder="Search for your signals..."
                )
            
            # 1b. Grab whatever IDs the user ALREADY selected (defaults to empty list on first load)
            # Using the widget's key lets us read the selection before the widget even renders!
            manage_search_selections = st.session_state.get(f"manage_signal_selector_{key_prefix}_{manage_form_id}", [])

            # 2. Filter your dataframe BEFORE feeding it to the multiselect!
            if manage_search.strip():
                manage_search_matches = signals_df[signals_df['search_text'].astype(str).str.contains(manage_search, case=False, na=False)]
            else:
                # Show only the 50 most recent signals by default if no search term is typed
                manage_search_matches = signals_df.head(50) 

            # 3. THE MAGIC: Pull the dataframe rows for anything currently selected
            manage_selected_matches = signals_df[signals_df['signal_id'].astype(str).isin(manage_search_selections)]
        
            # 4. Combine search results + already selected items, dropping any duplicates!
            manage_filtered_options = pd.concat([manage_search_matches, manage_selected_matches]).drop_duplicates(subset=['signal_id'])
                    
            # 3. Create a dictionary mapping ID -> Display Title for clean UI display
            manage_signal_map = {}
            for _, row in manage_filtered_options.iterrows():
                # Safely grab the UUID from your dataframe
                sig_id = str(row.get("signal_id", "Unknown-ID"))
                
                # Safely grab the header using your existing col_header variable
                # (If col_header isn't found, fall back to slicing search_text)
                if col_header and col_header in row and pd.notna(row[col_header]):
                    header = str(row[col_header])
                elif "search_text" in row and pd.notna(row["search_text"]):
                    header = str(row["search_text"])
                else:
                    header = "Untitled Signal"
                    
                # Truncate header to roughly one line of dropdown text
                short_header = header[:55] + "..." if len(header) > 55 else header
                
                # Safely grab the category/channel using your existing col_channel variable
                if col_channel and col_channel in row and pd.notna(row[col_channel]):
                    # If you have your display_channel_label function imported here, wrap it!
                    category = display_channel_label(str(row[col_channel]))
                else:
                    category = "UNCATEGORISED"
                    
                # --- CHOOSE YOUR FAVORITE DISPLAY FORMAT BELOW ---
                # Format A (Category + Header):
                label = f"[{category}] {short_header}"
                
                # Format B (Short Hex ID + Header - Uncomment line below if you prefer this!):
                # label = f"[#{sig_id[:6]}] {short_header}"
                
                manage_signal_map[sig_id] = label

            # 4. Render the clean multiselect tool
            multi_col, save_col = st.columns([4, 1])
            
            with multi_col:
                selected_to_attach = st.multiselect(
                    "Select Signals to Attach to this Cluster",
                    options=list(manage_signal_map.keys()), # Stores raw UUIDs in session state!
                    key=f"manage_signal_selector_{key_prefix}_{manage_form_id}",  # locks the widget to session state so that we can retrieve the selected options
                    format_func=lambda x: manage_signal_map.get(x, "Unknown Signal"), # Displays clean strings!
                    placeholder="Choose signals from the filtered list...", 
                    label_visibility="collapsed"
                )

            with save_col:
                if st.button("Attach Selected Signals", type="primary", width="stretch", disabled=(selected_to_attach==None), key=f"btn_attach_{key_prefix}_{target_cid}"):
                    if not selected_to_attach:
                        st.error("Please select signals before adding.")
                    else:
                        # TODO create function to add signals to cluster
                        # 1. Update Supabase
                        success, msg = add_signals_to_cluster(selected_to_attach, target_cid)
                        if success:
                            st.toast(msg)
                            # 2. Trigger AI Synthesis (Feature 3)
                            with st.spinner("🤖 Regenerating Evolving Synthesis..."):
                                success, msg = trigger_evolving_synthesis(current_id, signals_df)
                            if success:
                                st.toast(msg)
                            else:
                                st.error(msg)
                            load_cluster_graph_data.clear()
                            st.session_state['manage_form_id'] += 1
                            st.rerun()
                        else:
                            st.error(msg)

            st.divider()
            st.markdown("##### All Signals in Cluster")
            
            # --- BOTTOM: RENDER FULL CARDS ---
            if attached_signals_df.empty:
                st.info("No signals attached. Use the search tool above to bind intelligence to this theme.")
            else:
                # renders in rows of 3 like in the repository
                cluster_signal_rows = list(attached_signals_df.iterrows())
                for start in range(0, len(cluster_signal_rows), 3):
                    cols = st.columns(3)
                    for offset, (idx, row) in enumerate(cluster_signal_rows[start:start + 3]):
                        with cols[offset]:
                            render_signal_card(row, idx, key_prefix=f"cluster_expander_{target_cid}", cluster_map=master_cluster_map)

# -----------------------------
# Establish styles
# -----------------------------
st.markdown("""
    <style>
    /* =========================================================
       2. TARGETED EXPANDER FORMATTING
       ========================================================= */
    /* Theme-aware background and border styling */
    [data-testid="stExpander"]:has(.sticky-marker) {
        background-color: var(--background-color) !important;
        border: 1px solid var(--secondary-background-color) !important;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander title text formatting */
    [data-testid="stExpander"]:has(.sticky-marker) summary p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
    }

    /* ==========================================
       3. MULTISELECT FIXED HEIGHT + SCROLLING
       ========================================== */
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
        max-height: 80px !important;
        overflow-y: auto !important;
    }

    /* ==========================================
       4. SIDEBAR NAVIGATION BUTTONS
       ========================================== */
    /* Target buttons regardless of whether sidebar is a section, div, or aside */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 1.2rem !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
    }
    
    /* Force text INSIDE the button to align left and render as a prominent header */
    [data-testid="stSidebar"] [data-testid="stButton"] button p,
    [data-testid="stSidebar"] [data-testid="stButton"] button div {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        text-align: left !important;
        margin: 0 !important;
    }
            
    /* ==========================================
       5. LOCKED DROPDOWN MENU HEIGHT
       ========================================== */
    /* Target selectboxes with many options, prevents them from bleeding out of the box */
    div[data-baseweb="popover"] div[role="listbox"],
    ul[data-baseweb="menu"] {
        max-height: 260px !important; /* Shows ~6-7 rows before scrolling */
        overflow-y: auto !important;
    }
            
    /* =========================================================
       6. Styling for the sleek canvas navigation overlay
       ========================================================= */
    .canvas-nav-tip {
        position: absolute;
        top: 12px;
        right: 16px;
        background: rgba(140, 145, 155, 0.15);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: var(--text-color);
        border: 1px solid rgba(140, 145, 155, 0.25);
        z-index: 10;
        pointer-events: none; /* Lets analysts click and drag right through the text! */
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data for signals and clusters
# -----------------------------
signals_df = load_data()
if signals_df is None:
    st.error(f"Could not find {CSV_PATH}")
    st.stop()

clusters_df, cluster_signals_df, cluster_insights_df = load_cluster_graph_data()
master_cluster_map = dict(zip(clusters_df['id'].astype(str), clusters_df['title']))

# Flexible column matching
col_id = pick_column(signals_df, ["signal_id", "record_id"])
if col_id is None:
    def make_signal_id(row):
        raw = str(row.get("final_url", "") or row.get("link_url", "") or row.get("link/image", "") or row.get("scraped_header", ""))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    signals_df["signal_id"] = signals_df.apply(make_signal_id, axis=1)
    col_id = "signal_id"
col_time = pick_column(signals_df, ["message_time", "time of message"])
col_type = pick_column(signals_df, ["asset_type", "asset type"])
col_link = pick_column(signals_df, ["final_url", "link_url", "link/image"])
col_image = pick_column(signals_df, ["image_path", "image file", "local_image_path"])
col_header = pick_column(signals_df, ["scraped_header", "scraped header of the link"])
col_channel = pick_column(signals_df, ["sub_channel_name", "sub channel name"])
col_summary = None
col_tags = pick_column(signals_df, ["signal_hashtags", "article_hashtags", "llm_key_hashtags"])
col_discussion_tags = pick_column(signals_df, ["discussion_hashtags"])
col_extracted = pick_column(signals_df, ["article_text_extracted", "article_text_extracted?"])
col_stage = pick_column(signals_df, ["signal_stage", "suggested_stage", "stage"])
col_domain = pick_column(signals_df, ["source_domain"])
col_fetch_status = pick_column(signals_df, ["fetch_status"])
col_tag_origin = pick_column(signals_df, ["tag_origin"])
col_tag_review = pick_column(signals_df, ["tag_review_status"])

if col_type is None:
    signals_df["asset_type"] = "link"
    col_type = "asset_type"

if col_stage is None:
    signals_df["signal_stage"] = "NA"
    col_stage = "signal_stage"

# Merge human review scores into the main dataset.
votes_signals_df = vote_summary()
if col_id:
    signals_df[col_id] = signals_df[col_id].astype(str)
    signals_df = signals_df.merge(votes_signals_df, left_on=col_id, right_on="signal_id", how="left", suffixes=("", "_votes"))
else:
    signals_df["upvotes"] = 0
    signals_df["downvotes"] = 0
    signals_df["notes"] = 0
    signals_df["score"] = 0

for review_col in ["upvotes", "downvotes", "notes", "score"]:
    if review_col not in signals_df.columns:
        signals_df[review_col] = 0
    signals_df[review_col] = signals_df[review_col].fillna(0).astype(int)

# Merge hashtags added through the website into the main dataset.
user_tags_signals_df = user_tag_summary()
if col_id:
    signals_df = signals_df.merge(user_tags_signals_df, left_on=col_id, right_on="signal_id", how="left", suffixes=("", "_user_tags"))
else:
    signals_df["user_added_hashtags"] = ""
if "user_added_hashtags" not in signals_df.columns:
    signals_df["user_added_hashtags"] = ""
signals_df["user_added_hashtags"] = signals_df["user_added_hashtags"].fillna("")

# Downvotes act as vetoes: any veto pushes a record below all non-vetoed records.
signals_df["vetoed"] = signals_df["downvotes"] > 0
signals_df["opinion_rank"] = np.where(signals_df["vetoed"], -1_000_000 - signals_df["downvotes"], signals_df["upvotes"])

search_cols = [col_header, col_summary, col_tags, col_discussion_tags, col_channel, col_domain, "user_added_hashtags"]
signals_df["search_text"] = signals_df.apply(lambda row: build_search_text(row, search_cols), axis=1)

signals_df["parsed_hashtags"] = signals_df.apply(build_combined_hashtags, axis=1)
signals_df["has_no_hashtag"] = signals_df["parsed_hashtags"].apply(lambda tags: len(tags) == 0)

all_tags = sorted(set(tag for tags in signals_df["parsed_hashtags"] for tag in tags), key=str.lower)

signals_df["message_dt"] = pd.to_datetime(signals_df[col_time], errors="coerce") if col_time else pd.NaT

with st.spinner("Preparing semantic search and clusters..."):
    embeddings = compute_embeddings(signals_df["search_text"].fillna("").tolist())
    signals_df = add_cluster_labels(signals_df, embeddings, n_clusters=8)
    signals_df = label_clusters(signals_df, col_tags, col_header)

# ============================
# Rendering starts here
# ============================
# STATE INITIALISATION
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# LOGIN PAGE GATEKEEPER
if not st.session_state['current_user']:
    st.title("CSF Horizon Scanning Platform")
    st.write("Who is driving today?")
    
    selected_name = st.selectbox(
        "Analyst Name", 
        options=ANALYSTS, 
        index=None,     # leaves the dropdown blank
        placeholder="Type or select your name..."  
        )
    
    if st.button("Enter Workspace →", type="primary", disabled=(selected_name==None)):
        if selected_name:
            st.session_state['current_user'] = selected_name
            st.session_state['active_page'] = "Signal Repository"
            st.rerun()
        else:
            st.warning("Please select your name from the list to log in.")
            
    st.stop() # <-- FATAL STOP: Prevents Streamlit from rendering ANY code below this line!

# ==========================================
# 0. INITIALIZE SESSION STATE (Page Memory)
# ==========================================
# active_page is stored in session_state.
# This ensures Streamlit remembers the active page even when you interact with filters. This also makes the homepage Signal Repository
if 'active_page' not in st.session_state:
    st.session_state['active_page'] = "Signal Repository"

# ==========================================
# 1. SIDEBAR WITH INDIVIDUAL BUTTONS
# ==========================================
with st.sidebar:
    st.header("CSF Horizon Scanning Platform")
    st.write(f"**Active User:** {get_current_user()}")
    if st.button("👥 Change User", width='stretch'):
        st.session_state["current_user"] = None
        st.rerun()

    if st.button("🔄 Refresh Data", width='stretch'):
        load_cluster_graph_data.clear()
        st.rerun()

    st.divider()

    # We use width='stretch' so buttons stretch nicely across the whole sidebar.
    # We dynamically change type="primary" to highlight whichever button is currently active!
    if st.button("Ingestion Hub", 
                 width='stretch', 
                 type="primary" if st.session_state['active_page'] == "Ingestion Hub" else "secondary"):
        st.session_state['active_page'] = "Ingestion Hub"
        st.session_state['current_visited_page'] = 'ingestion_hub'    # for state memory so that it can reset the cluster view in cluster bank
        st.rerun()  # Forces an immediate clean reload of the page
    
    if st.button("Signal Repository", 
                 width='stretch', 
                 type="primary" if st.session_state['active_page'] == "Signal Repository" else "secondary"):
        st.session_state['active_page'] = "Signal Repository"
        st.session_state['current_visited_page'] = 'signal_repository'    # for state memory so that it can reset the cluster view in cluster bank
        st.rerun()
        
    if st.button("Cluster Bank", 
                 width='stretch', 
                 type="primary" if st.session_state['active_page'] == "Cluster Bank" else "secondary"):
        st.session_state['active_page'] = "Cluster Bank"
        st.rerun()

    if st.button("Analytics Dashboard",
                 width='stretch',
                 type="primary" if st.session_state['active_page'] == "Analytics Dashboard" else "secondary"):
        st.session_state['active_page'] = "Analytics Dashboard"
        st.session_state['current_visited_page'] = 'analytics_dashboard'    # for state memory so that it can reset the cluster view in cluster bank
        st.rerun()

# ==========================================
# 2. PAGE ROUTING (Reading from Memory)
# ==========================================
# Below contains all the information for each page


# 2a. Ingestion Hub Page
# ==========================================
if st.session_state['active_page'] == "Ingestion Hub":
    st.title("Ingestion Hub")
    st.info("🚧 Work in progress...")

# 2b. Signal Dashboard Page
# ==========================================
elif st.session_state['active_page'] == "Signal Repository":
    # --- Your Existing Dashboard Code Goes Here ---
    st.title("Signal Repository")

    with st.expander("Search and Filter Signals", expanded=True):
        
        # ---> THE SECRET MARKER (Invisible HTML div) <---
        # This 1 line is what tells our CSS above to freeze THIS specific expander!
        st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)

        # --- ROW 1: Keyword (2 cols) & Semantic Similarity (1 col) ---
        col1, col2 = st.columns([2, 1]) 
        with col1:
            keyword_query = st.text_input("Keyword Search", placeholder="Type keywords to filter...")
        with col2:
            semantic_query = st.text_input("Semantic Similarity", placeholder="Describe concept...")

        # --- ROW 2: Asset Type, Category (by whatsapp chat), and Hashtag (3 equal cols) ---
        col3, col4, col5 = st.columns(3)
        with col3:
            # Creates list of asset types from database
            asset_types = sorted(signals_df[col_type].dropna().astype(str).unique().tolist())
            # Multiselect tool
            asset_type_filter = st.multiselect("Asset Type", options=asset_types, default=asset_types, placeholder="Select types...")
        with col4:
            # Pulls channel labels and makes them into categories
            if col_channel:
                channels = sorted(
                    signals_df[col_channel].dropna().astype(str).unique().tolist(),
                    key=lambda value: display_channel_label(value),
                )
                # selected_channels is the multiselect tool
                category_filter = st.multiselect(
                    "Category",
                    channels,
                    default=channels,
                    format_func=display_channel_label,
                    placeholder="Select categories..."
                )
            else:   # if no channels in database, then leaves this empty
                category_filter = None            
        with col5:
            hashtag_options = ["No hashtag (N/A)"] + all_tags
            hashtag_filter = st.multiselect("Hashtags", options=hashtag_options, default=[], placeholder="Select tags...")            

        # --- ROW 3: Date Range & Future Cluster Features (3 equal cols) ---
        col6, col7, col8, col9 = st.columns([2, 2, 1, 1])
        with col6:
            # checks if database has date values. If yes, pulls the min and max date as a range for filtering
            if signals_df["message_dt"].notna().any():
                min_date = signals_df["message_dt"].min().date()
                max_date = signals_df["message_dt"].max().date()
                # date input tool
                date_filter = st.date_input("Signal Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_filter = None
            
        with col7:
            # --- CLUSTER SEARCH FEATURE ---
            # Create a mapping dictionary of UUID -> Title
            cluster_search_map = dict(zip(clusters_df['id'].astype(str), clusters_df['title']))

            cluster_filter = st.multiselect(
                "Filter by Cluster", 
                options=list(cluster_search_map.keys()), 
                format_func=lambda cid: f"{cluster_search_map.get(cid, 'Unknown Cluster')}",
                placeholder="Type keyword to select multiple clusters...",
                default=[]
                )
            
        with col8:
            # --- CLUSTER SEARCH FEATURE ---
            st.write("") # Adds vertical spacing to align checkbox nicely with dropdowns
            st.write("")
            show_clustered = st.checkbox("Show Clustered Signals", value=True)

        with col9:
            # --- CLUSTER SEARCH FEATURE ---
            st.write("") # Adds vertical spacing to align checkbox nicely with dropdowns
            st.write("")
            show_unclustered = st.checkbox("Show Unclustered Signals", value=True)

    filtered = signals_df.copy()
    filtered = filtered[filtered[col_type].astype(str).isin(asset_type_filter)]

    if col_channel and category_filter is not None:
        filtered = filtered[filtered[col_channel].astype(str).isin(category_filter)]

    if date_filter and signals_df["message_dt"].notna().any():
        if isinstance(date_filter, tuple) and len(date_filter) == 2:
            start_date, end_date = date_filter
        else:
            start_date = end_date = date_filter
        filtered = filtered[
            (filtered["message_dt"].dt.date >= start_date)
            & (filtered["message_dt"].dt.date <= end_date)
        ]

    if hashtag_filter:
        selected_real_tags = {tag.lower() for tag in hashtag_filter if tag != "No hashtag (N/A)"}
        include_no_tag = "No hashtag (N/A)" in hashtag_filter

        def hashtag_filter(tags):
            tag_set = {tag.lower() for tag in tags}
            return bool(tag_set & selected_real_tags) or (include_no_tag and not tag_set)

        filtered = filtered[filtered["parsed_hashtags"].apply(hashtag_filter)]

    if keyword_query:
        filtered = filtered[
            (filtered["search_text"].astype(str).str.contains(keyword_query, case=False, na=False)) 
        ]

    if semantic_query.strip():
        query_embedding = compute_embeddings([semantic_query])[0]
        sims = cosine_similarity([query_embedding], embeddings)[0]
        signals_df["semantic_score"] = sims
    else:
        signals_df["semantic_score"] = np.nan

    # 1. Identify all signal IDs currently attached to AT LEAST ONE cluster
    #    We use a set() for O(1) lookup speed
    clustered_signal_ids = set(cluster_signals_df['signal_id'].astype(str)) if not cluster_signals_df.empty else set()
    
    # 2. APPLY MULTISELECT FILTER (Specific Clusters)
    #    If the analyst chose specific clusters in the dropdown, narrow down to those exact signals
    if cluster_filter:
        matched_links = cluster_signals_df[cluster_signals_df['cluster_id'].astype(str).isin(cluster_filter)]
        target_signal_ids = set(matched_links['signal_id'].astype(str))
        filtered = filtered[filtered['signal_id'].astype(str).isin(target_signal_ids)]

    # 3. APPLY CHECKBOX FILTERS (Clustered vs. Unclustered Status)
    if not show_clustered and not show_unclustered:
        # State 1: Both unticked -> Show NOTHING
        filtered = filtered.iloc[0:0]  # Returns empty DF preserving schema

    elif show_clustered and not show_unclustered:
        # State 2: Only Clustered -> Keep rows whose ID is IN our clustered set
        filtered = filtered[filtered['signal_id'].astype(str).isin(clustered_signal_ids)]

    elif not show_clustered and show_unclustered:
        # State 3: Only Unclustered -> Keep rows whose ID is NOT IN our clustered set (~ performs logical NOT)
        filtered = filtered[~filtered['signal_id'].astype(str).isin(clustered_signal_ids)]

    # State 4 (Both ticked): Skip filtering entirely, leaving filtered_signals_df as-is!

    # # Remove these temporarily to streamline results page
    # c1, c2, c3, c4 = st.columns(4)
    # c1.metric("Matching records", total_matching)
    # c2.metric("Total records", len(signals_df))
    # c3.metric("Total unique hashtags", len(all_tags))
    # c4.metric("Positive votes", int(signals_df["upvotes"].sum()))

    # Pagination: the result set is no longer capped with .head().
    # Instead, all matching records are split into pages.
    
    # removed temporarily until figure out how to do custom no. of results. Unless not necessary
    # page_size = st.selectbox("Records per page", [9, 18, 27, 36, 54], index=1)

    # ==========================================
    # SETUP DATA & STATE MEMORY
    # ==========================================
    # Setup the number of pages of results based on matched records
    total_matching = len(filtered)
    page_size = 18
    total_pages = max(1, int(np.ceil(total_matching / page_size)))

    # Ensure it always starts on page 1
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1
    if st.session_state["current_page"] > total_pages:
        st.session_state["current_page"] = 1
    if st.session_state["current_page"] < 1:
        st.session_state["current_page"] = 1

    # ==========================================
    # DYNAMIC PAGINATION BUTTONS (Row 1)
    # ==========================================
    st.divider()
    render_pagination(total_pages, key_prefix="top")

    # ==========================================
    # SORTING & STATS HEADER (Row 2)
    # ==========================================
    header_left, header_right = st.columns([2, 1])

    # shows how many results per page and how many pages. Takes up 2 columns out of 3
    with header_left:
        # Calculate exact indices for the "n out of m" display
        start_idx = (st.session_state['current_page'] - 1) * page_size
        end_idx = min(start_idx + page_size, total_matching)
        
        if total_matching > 0:
            st.markdown(f"**Showing results {start_idx + 1}–{end_idx}** of **{total_matching}** *(Page {st.session_state['current_page']} of {total_pages})*")
        else:
            st.warning("No signals found matching your search.")

    with header_right:
        # Single combined sort selectbox
        sort_option = st.selectbox(
            "Sort by",
            options=[ 
                "Newest First", 
                "Oldest First", 
                "Alphabetical (A-Z)", 
                "Alphabetical (Z-A)",
                "Relevance (Semantic)"
            ],
            label_visibility="collapsed" # Hides the label so it aligns nicely with the text on the left
        )

    # TODO --- APPLY SORTING LOGIC ---
    if sort_option == "Newest First":
        filtered = filtered.sort_values(by="message_dt", ascending=False)
    elif sort_option == "Oldest First":
        filtered = filtered.sort_values(by="message_dt", ascending=True)
    elif sort_option == "Alphabetical (A-Z)":
        filtered = filtered.sort_values(by="search_text", ascending=True)
    elif sort_option == "Alphabetical (Z-A)":
        filtered = filtered.sort_values(by="search_text", ascending=False)
    elif sort_option == "Relevance (Semantic)" and semantic_query.strip():
        # Ensure your search function added a 'semantic_score' column!
        filtered = filtered.loc[signals_df.loc[filtered.index, "semantic_score"].sort_values(ascending=False).index]

    # # Remove these temporarily to streamline results page
    # st.markdown("### Top hashtag pairs in matching records")
    # st.caption(
    #     "Counts how many matching records contain both hashtags together. Each record counts once per pair."
    # )
    # pair_counts = hashtag_pair_frame(filtered, top_n=5)
    # if pair_counts.empty:
    #     st.write("Not enough co-occurring hashtags in the current matching records.")
    # else:
    #     st.dataframe(pair_counts, width='stretch', hide_index=True)

    # TODO results page
    page_df = filtered.iloc[start_idx:end_idx]

    st.markdown("## Results")
    if page_df.empty:
        st.info("No records match the current filters.")
    else:
        rows = list(page_df.iterrows())
        for start in range(0, len(rows), 3):
            cols = st.columns(3)
            for offset, (idx, row) in enumerate(rows[start:start + 3]):
                with cols[offset]:
                    render_signal_card(row, idx, semantic_query=semantic_query, key_prefix="repo", cluster_map=master_cluster_map)

    # ==========================================
    # DYNAMIC PAGINATION BUTTONS (Last Row)
    # ==========================================
    st.divider()
    render_pagination(total_pages, key_prefix="bottom")

# 2c. Cluster Bank Page
# ==========================================
elif st.session_state['active_page'] == "Cluster Bank":
    st.title("Cluster Bank")
    
    # ==========================================
    # # CREATE CLUSTER FUNCTION
    # ==========================================
    with st.expander("Create New Cluster", expanded=False):

        st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)    # same formatting style as the signal repository search expander

        if 'cluster_form_id' not in st.session_state:
            st.session_state['cluster_form_id'] = 0

        form_id = st.session_state['cluster_form_id']

        col1, col2 = st.columns([3, 1])
        with col1:
            new_title = st.text_input(
                "Cluster Title", 
                key=f"cluster_title_input_{form_id}", 
                placeholder="Give it a catchy name. You can always change this later!"
                )
        with col2:
            new_status = st.selectbox(
                "Maturity Status", 
                options=["Active", "KIV", "Presentation-ready", "Archived"], 
                key=f"cluster_status_input_{form_id}"
                )
            
        st.markdown("#### Attach Existing Signals *(Optional)*")
        
        # 1a. Provide a quick keyword filter so analysts don't have to scroll through hundreds of rows
        search_filter = st.text_input(
            "Filter signals by keyword and select them below", 
            key=f"cluster_signal_search_{form_id}", 
            placeholder="Search for your signals...")
        
        # 1b. Grab whatever IDs the user ALREADY selected (defaults to empty list on first load)
        # Using the widget's key lets us read the selection before the widget even renders!
        current_selections = st.session_state.get(f"cluster_signal_selector_{form_id}", [])

        # 2. Filter your dataframe BEFORE feeding it to the multiselect!
        if search_filter.strip():
            search_matches = signals_df[signals_df['search_text'].astype(str).str.contains(search_filter, case=False, na=False)]
        else:
            # Show only the 50 most recent signals by default if no search term is typed
            search_matches = signals_df.head(50) 

        # 3. THE MAGIC: Pull the dataframe rows for anything currently selected
        selected_matches = signals_df[signals_df['signal_id'].astype(str).isin(current_selections)]
    
        # 4. Combine search results + already selected items, dropping any duplicates!
        filtered_options = pd.concat([search_matches, selected_matches]).drop_duplicates(subset=['signal_id'])
                
        # 3. Create a dictionary mapping ID -> Display Title for clean UI display
        signal_map = {}
        for _, row in filtered_options.iterrows():
            # Safely grab the UUID from your dataframe
            sig_id = str(row.get("signal_id", "Unknown-ID"))
            
            # Safely grab the header using your existing col_header variable
            # (If col_header isn't found, fall back to slicing search_text)
            if col_header and col_header in row and pd.notna(row[col_header]):
                header = str(row[col_header])
            elif "search_text" in row and pd.notna(row["search_text"]):
                header = str(row["search_text"])
            else:
                header = "Untitled Signal"
                
            # Truncate header to roughly one line of dropdown text
            short_header = header[:55] + "..." if len(header) > 55 else header
            
            # Safely grab the category/channel using your existing col_channel variable
            if col_channel and col_channel in row and pd.notna(row[col_channel]):
                # If you have your display_channel_label function imported here, wrap it!
                category = display_channel_label(str(row[col_channel]))
            else:
                category = "UNCATEGORISED"
                
            # --- CHOOSE YOUR FAVORITE DISPLAY FORMAT BELOW ---
            # Format A (Category + Header):
            label = f"[{category}] {short_header}"
            
            # Format B (Short Hex ID + Header - Uncomment line below if you prefer this!):
            # label = f"[#{sig_id[:6]}] {short_header}"
            
            signal_map[sig_id] = label

        # 4. Render the clean multiselect tool
        selected_signal_ids = st.multiselect(
            "Select Signals to Attach",
            options=list(signal_map.keys()), # Stores raw UUIDs in session state!
            key=f"cluster_signal_selector_{form_id}",  # locks the widget to session state so that we can retrieve the selected options
            format_func=lambda x: signal_map.get(x, "Unknown Signal"), # Displays clean strings!
            placeholder="Choose signals from the filtered list...", 
            label_visibility="collapsed"
        )
        
        # 5. Create cluster button
        if st.button("Create Cluster", type="primary", width='stretch', disabled=(new_title==None)):
            if not new_title.strip():
                st.error("Please provide a cluster title before saving.")
            else:
                # run save_new_cluster() to save to Supabase
                with st.spinner("Saving cluster to Supabase..."):
                    success, message = save_new_cluster(
                        title=new_title, 
                        status=new_status, 
                        signal_ids=selected_signal_ids
                        )
                if success:
                    st.session_state['cluster_success_msg'] = message
                    
                    # Clear cache so that newly created cluster appears in node graph
                    load_cluster_graph_data.clear()
                    # and update widget key
                    st.session_state['cluster_form_id'] += 1

                    st.rerun()
                else:
                    st.error(message)

        if 'cluster_success_msg' in st.session_state:
            st.toast(f"✅ {st.session_state['cluster_success_msg']}")
            del st.session_state['cluster_success_msg']


    # ==========================================
    # CLUSTER NODE GRAPH SECTION
    # ==========================================
    # Search functions to filter clusters

    # UI CONTROLS
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_cluster_ids = st.multiselect(
        "Search and Select Clusters",
        options=list(master_cluster_map.keys()),
        format_func=lambda cid: f"{master_cluster_map.get(cid, 'Unknown Cluster')}",
        placeholder="Type keyword to select multiple clusters...",
        default=None
    )
    with col2:
        maturity_filter = st.multiselect(
            "Maturity", 
            ["Active", "KIV", "Presentation-ready", "Archived"], 
            default=["Active", "KIV", "Presentation-ready"], 
            placeholder="Filter by maturity..."
            )
    with col3:
        st.write("")
        st.write("")        
        isolate_mode = st.checkbox("Isolate View Mode", value=True)

    # PYTHON DATA PIPELINE (Builds the relations before sending it to ECharts to increase performance)
    # 1. Filter by Maturity Status first
    active_clusters = clusters_df[clusters_df['maturity_status'].isin(maturity_filter)]

    # 2. Build Nodes and Edges arrays for ECharts
    nodes, edges = build_cluster_topology(active_clusters, cluster_signals_df, signals_df, selected_cluster_ids, isolate_mode)

    # Render the graph canvas
    with st.container(border=True):
        # 1. Inject the CSS grid marker AND the semi-transparent navigation overlay!
        st.markdown("""
                    <div class="canvas-nav-tip">
                    💡 <b>Canvas Controls:</b> Click & drag background to pan • Scroll or pinch to zoom • Click on nodes to view details
                    </div>
                    """, unsafe_allow_html=True)
        
        # --- ECHARTS ANIMATED PAYLOAD ---
        graph_options = {
            "backgroundColor": "rgba(140, 145, 155, 0.08)",

            "animationDurationUpdate": 1000,  # 1-second smooth sliding/fading animation!
            "animationEasingUpdate": "quinticInOut",
            "series": [{
                "type": "graph",
                "layout": "force",
                "data": nodes,
                "links": edges,
                "roam": True,  # Allows users to click-and-drag pan or mousewheel zoom

                "force": {
                    "repulsion": 500,
                    "edgeLength": [120, 220],
                    "gravity": 0.04
                },
                # EMPHASIS: What happens when a user hovers or when a node is highlighted
                "emphasis": {
                    "focus": "adjacency",  # Automatically dims all non-connected nodes!
                    "lineStyle": { "width": 3, "color": "#1C83E1" }
                }
            }]
        }
        
        # The event handler intercepts the browser click and sends a clean dict to Python
        clicked_node = st_echarts(
            options=graph_options, 
            height="650px",
            key="cluster_overview_graph", 
            events={"click": "function(params) { return params.data; }"}
        )

    # =========================================================
    # ROUTING CONTROLLER TO CLUSTER / SIGNAL DETAILS
    # =========================================================
    # If the analyst just navigated here from another page, wipe old inspection memory!
    # Navigating to other pages changes st.session_state['current_visited_page'] to another value. Coming back here will keep that value, and only when memory is reset, then we update the value again.
    if st.session_state.get('current_visited_page') != 'cluster_bank':
        st.session_state['active_view_type'] = None
        st.session_state['active_view_id'] = None
        st.session_state['secondary_cluster_id'] = None
        st.session_state['trigger_scroll'] = False
        st.session_state['current_visited_page'] = 'cluster_bank'

    # st.write("Raw Click Payload:", clicked_node)      # 🐞 TEMPORARY DEBUG: Uncomment this line to see EXACTLY what Python receives!

    event_data = clicked_node.get('chart_event', {}) if isinstance(clicked_node, dict) else {}
    if event_data and event_data.get('id'):
        new_id = str(event_data.get('id')).strip().lower()

        # KIV AUTOSCROLL FUNCTION
        if new_id != st.session_state.get('active_view_id'):
            # Trigger scroll anchor flag
            st.session_state['trigger_scroll'] = True

        st.session_state['active_view_type'] = event_data.get('node_type')
        st.session_state['active_view_id'] = new_id
        # Reset secondary view when a new node is clicked on the canvas
        st.session_state['secondary_cluster_id'] = None

    # Route from persistent state
    # KIV AUTOSCROLL FUNCTION
    st.markdown("<div id='inspection_panel_anchor'></div>", unsafe_allow_html=True)

    current_type = st.session_state['active_view_type']
    current_id = st.session_state['active_view_id']

    if not current_id:
        st.info("👆 **Select a node:** Click any cluster or signal dot on the map above to inspect its details.")

    elif current_type == 'signal':
        # --- SIGNAL INSPECTION MODE ---
        target_sig_id = current_id
        sig_row = signals_df[signals_df['signal_id'].astype(str).str.strip().str.lower() == target_sig_id]
        
        if sig_row.empty:
            st.error(f"❌ Error: Node ID `{target_sig_id}` was clicked, but no matching UUID exists in `signals_df`!")
        else:
            st.markdown(f"### 🔎 Inspecting Signal: `#{target_sig_id[:6]}`")
            # 1. Show Parent Cluster Badges
            parent_links = cluster_signals_df[cluster_signals_df['signal_id'].astype(str) == target_sig_id]
            parent_cids = parent_links['cluster_id'].astype(str).tolist()
            parent_clusters = clusters_df[clusters_df['id'].astype(str).isin(parent_cids)]
            
            # The badges show up as cool small cards that are clickable to then generate the cluster data below without making this part disappear.
            # So that it's the same as if you clicked the cluster node
            # The signal details disappear ONLY if they click another node in the graph
            st.write("**Connected Strategic Themes:** ")
            chip_cols = st.columns(len(parent_clusters) if len(parent_clusters) > 0 else 1)

            for idx, (_, c_row) in enumerate(parent_clusters.iterrows()):
                cid_str = str(c_row['id'])
                with chip_cols[idx]:
                    if st.button(f"🎯 {c_row['title']}", key=f"chip_{cid_str}", use_container_width=True):
                        # Toggle the secondary cluster view!
                        st.session_state['secondary_cluster_id'] = cid_str
                        st.rerun()

            st.markdown("")
            
            # 2. Render signal card function, check that it matches the function arguments
            render_signal_card(sig_row.iloc[0], idx=0, key_prefix="inspection", cluster_map=master_cluster_map)

            if st.session_state['secondary_cluster_id']:
                st.markdown("---")
                st.markdown("### 🎯 Parent Cluster Overview")
                render_cluster_dashboard(st.session_state['secondary_cluster_id'], clusters_df, cluster_signals_df, signals_df, key_prefix="secondary")

    elif current_type == 'cluster':
        # --- CLUSTER DASHBOARD MODE ---
        render_cluster_dashboard(current_id, clusters_df, cluster_signals_df, signals_df, key_prefix="primary")

    else:
        # ---> THE CATCH-ALL: Prevents blank screens if data is malformed! <---
        st.warning(f"⚠️ Unrecognized node click! Python received: `{clicked_node}`")
        st.caption("Tip: Check `build_cluster_topology` to ensure every node dictionary explicitly includes `'node_type': 'cluster'` or `'node_type': 'signal'`.")

    # AUTO-SCROLL TO CLUSTER/SIGNAL DISPLAY BELOW GRAPH VIEW
    # KIV because can't nail down the mechanics yet
    if st.session_state.get('trigger_scroll', False):
        st.components.v1.html("""
            <script>
                const parentDoc = window.parent.document;
                const anchor = parentDoc.getElementById('inspection_panel_anchor');
                if (anchor) {
                    anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            </script>
        """, height=0)
        
        # Reset flag so it only scrolls once when a new node is clicked!
        st.session_state['trigger_scroll'] = False

# 2d. Analytics Dashboard (Taken from previous "overview" tab)
# ==========================================
elif st.session_state['active_page'] == "Analytics Dashboard":
    st.title("Analytics Dashboard")
    st.markdown("## Overview")

    if col_time and signals_df["message_dt"].notna().any():
        latest_time = signals_df["message_dt"].max()
        recent_cutoff = latest_time - pd.Timedelta(days=30)
        recent = signals_df[signals_df["message_dt"] >= recent_cutoff].copy()
    else:
        latest_time = None
        recent = signals_df.copy()

    timeframe = st.radio(
        "Overview timeframe",
        ["All time", "Last 30 days"],
        horizontal=True,
    )
    overview_df = signals_df.copy() if timeframe == "All time" else recent.copy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signals shown", len(overview_df))
    m2.metric("All records", len(signals_df))
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
        st.dataframe(top_reviewed[display_cols], width='stretch', hide_index=True)

    if col_fetch_status:
        st.markdown("### Fetch status mix")
        fetch_df = count_frame(overview_df, col_fetch_status, "fetch_status", top_n=20, include_na=True)
        st.dataframe(fetch_df, width='stretch', hide_index=True)