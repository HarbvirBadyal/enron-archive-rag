# Import the app dependencies and resolve project files
import html
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
EMAILS_FILE = APP_DIRECTORY / "enron_clean.csv"

# Use the project directory as the working directory for backend model and index paths.
os.chdir(APP_DIRECTORY)

# Configure the Streamlit page title, layout, icon, and sidebar state.
st.set_page_config(
    page_title="Enron Archive Search",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Apply the custom visual theme for the search interface, results, source cards, and loading states.
st.markdown(
    """
    <style>
    :root {
        --ink: #eef4ff;
        --muted: #90a0bd;
        --panel: rgba(12, 20, 39, 0.78);
        --panel-strong: rgba(17, 28, 52, 0.94);
        --line: rgba(150, 177, 225, 0.16);
        --blue: #6ea8ff;
        --cyan: #65e4dc;
        --violet: #9c86ff;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
            "Segoe UI", sans-serif;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 50% -12%, rgba(67, 113, 205, 0.30), transparent 34rem),
            radial-gradient(circle at 8% 82%, rgba(60, 213, 195, 0.09), transparent 26rem),
            linear-gradient(160deg, #08101f 0%, #0a1325 52%, #08101d 100%);
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"], footer {
        visibility: hidden;
    }

    .block-container {
        width: min(100%, 54rem);
        padding-top: 3.2rem;
        padding-bottom: 5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.8rem;
        margin-bottom: 2rem;
        user-select: none;
    }

    .brand-mark {
        position: relative;
        width: 2.25rem;
        height: 2.25rem;
        border-radius: 0.78rem;
        background: linear-gradient(145deg, var(--cyan), var(--blue) 55%, var(--violet));
        box-shadow: 0 0.8rem 2.3rem rgba(86, 153, 255, 0.28);
        transform: rotate(-8deg);
    }

    .brand-mark::after {
        content: "";
        position: absolute;
        width: 0.48rem;
        height: 0.48rem;
        top: 0.48rem;
        right: 0.48rem;
        border-radius: 50%;
        background: #ffffff;
        box-shadow: -0.45rem 0.52rem 0 -0.08rem rgba(255,255,255,0.7);
    }

    .brand-name {
        color: #f6f9ff;
        font-size: 1.28rem;
        font-weight: 650;
        letter-spacing: -0.025em;
    }

    .brand-name span {
        color: var(--muted);
        font-weight: 450;
    }

    .welcome {
        text-align: center;
        margin: 3.1rem auto 1.7rem;
    }

    .welcome h1 {
        margin: 0;
        color: #f4f7ff;
        font-size: clamp(2rem, 6vw, 3.35rem);
        line-height: 1.05;
        letter-spacing: -0.055em;
        font-weight: 660;
    }

    .welcome p {
        margin: 0.9rem auto 0;
        color: var(--muted);
        font-size: 1rem;
    }

    [data-testid="stForm"] {
        padding: 0.45rem 0.5rem 0.45rem 1rem;
        border: 1px solid rgba(138, 173, 230, 0.22);
        border-radius: 1.35rem;
        background: rgba(15, 25, 47, 0.88);
        box-shadow: 0 1.4rem 4rem rgba(0, 0, 0, 0.27),
            inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(18px);
    }

    [data-testid="stForm"] > div {
        gap: 0.5rem;
    }

    [data-testid="stTextInput"] input {
        min-height: 3.35rem;
        border: 0 !important;
        color: #f4f7ff !important;
        background: transparent !important;
        box-shadow: none !important;
        font-size: 1.02rem;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #70809d;
    }

    [data-testid="stFormSubmitButton"] button {
        min-height: 3.15rem;
        padding: 0 1.35rem;
        border: 0;
        border-radius: 1rem;
        color: #06101f;
        background: linear-gradient(135deg, var(--cyan), var(--blue));
        font-weight: 700;
        box-shadow: 0 0.65rem 1.5rem rgba(79, 151, 255, 0.20);
        transition: transform 150ms ease, filter 150ms ease;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        color: #06101f;
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    .history-title {
        margin: 2.8rem 0 0.9rem;
        color: #7788a7;
        font-size: 0.77rem;
        font-weight: 720;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 1.25rem !important;
        background: linear-gradient(145deg, rgba(18, 29, 53, 0.82), rgba(11, 19, 36, 0.88));
        box-shadow: 0 1rem 2.5rem rgba(0,0,0,0.16);
    }

    .question-line {
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
        margin-bottom: 1rem;
        color: #c7d3e8;
        font-size: 0.96rem;
    }

    .question-dot {
        width: 0.48rem;
        height: 0.48rem;
        margin-top: 0.42rem;
        flex: 0 0 auto;
        border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 0.8rem rgba(101, 228, 220, 0.55);
    }

    .answer-copy {
        color: #f4f7ff;
        font-size: 1.22rem;
        line-height: 1.55;
        letter-spacing: -0.015em;
    }

    .answer-meta {
        margin-top: 0.85rem;
        color: #73839f;
        font-size: 0.78rem;
    }

    .stButton button {
        border: 1px solid rgba(111, 167, 255, 0.28);
        border-radius: 0.85rem;
        color: #bcd7ff;
        background: rgba(38, 73, 124, 0.18);
        font-weight: 620;
        transition: border-color 150ms ease, background 150ms ease;
    }

    .stButton button:hover {
        border-color: rgba(101, 228, 220, 0.55);
        color: #dffdfa;
        background: rgba(44, 115, 124, 0.18);
    }

    .source-card {
        margin-top: 0.55rem;
        padding: 1.25rem;
        border: 1px solid rgba(101, 228, 220, 0.17);
        border-radius: 1rem;
        background: rgba(5, 13, 27, 0.60);
    }

    .source-kicker {
        color: var(--cyan);
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .source-subject {
        margin: 0.45rem 0 1rem;
        color: #eef4ff;
        font-size: 1.08rem;
        font-weight: 650;
    }

    .source-grid {
        display: grid;
        grid-template-columns: 5rem 1fr;
        gap: 0.45rem 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--line);
        color: #b9c6dc;
        font-size: 0.84rem;
        overflow-wrap: anywhere;
    }

    .source-label {
        color: #657793;
        font-weight: 620;
    }

    .email-body {
        max-height: 28rem;
        margin-top: 1rem;
        padding-right: 0.45rem;
        overflow-y: auto;
        color: #cbd5e7;
        font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
        font-size: 0.84rem;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .loader-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 10rem;
        gap: 1rem;
    }

    .comet-loader {
        position: relative;
        width: 3.4rem;
        height: 3.4rem;
        animation: comet-orbit 1.05s linear infinite;
    }

    .comet-loader::before {
        content: "";
        position: absolute;
        top: 0.15rem;
        left: 50%;
        width: 0.58rem;
        height: 0.58rem;
        border-radius: 50%;
        background: #d9fffb;
        box-shadow:
            -0.22rem 0.12rem 0.18rem rgba(101, 228, 220, 0.78),
            -0.5rem 0.24rem 0.32rem rgba(101, 228, 220, 0.48),
            -0.84rem 0.34rem 0.48rem rgba(110, 168, 255, 0.24);
    }

    .loader-label {
        color: #8091ae;
        font-size: 0.86rem;
        letter-spacing: 0.02em;
    }

    @keyframes comet-orbit {
        to { transform: rotate(360deg); }
    }

    @media (max-width: 640px) {
        .block-container { padding: 2rem 1rem 4rem; }
        .welcome { margin-top: 2.2rem; }
        [data-testid="stForm"] { padding-left: 0.7rem; }
        [data-testid="stFormSubmitButton"] button { padding: 0 0.95rem; }
        .source-grid { grid-template-columns: 4.2rem 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and cache the final RAG question-answering function for reuse across Streamlit reruns.
@st.cache_resource(show_spinner=False)
def get_answer_question():
    from generation_final import answer_question

    return answer_question

# Find and cache a cited source email by scanning the cleaned archive in small batches.
@st.cache_data(show_spinner=False)
def find_email(email_id):
    columns = [
        "email_id",
        "date",
        "sender",
        "recipients",
        "subject",
        "body",
    ]

    for email_batch in pd.read_csv(
        EMAILS_FILE,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=5_000,
    ):
        match = email_batch[email_batch["email_id"] == email_id]

        if not match.empty:
            return match.iloc[0].to_dict()

    return None

# Escape dynamic text before inserting it into custom HTML and provide a fallback for missing values.
def safe_text(value, fallback="Not provided"):
    value = str(value or "").strip()
    return html.escape(value) if value else fallback

# Build the reusable loading animation shown while searches and source lookups are running.
def loading_animation(message):
    return f"""
    <div class="loader-wrap" role="status" aria-live="polite">
        <div class="comet-loader" aria-hidden="true"></div>
        <div class="loader-label">{html.escape(message)}</div>
    </div>
    """

# Render the cited original email and its metadata in a traceable source card.
def render_source(email):
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-kicker">Original email</div>
            <div class="source-subject">{safe_text(email.get("subject"), "No subject")}</div>
            <div class="source-grid">
                <div class="source-label">From</div>
                <div>{safe_text(email.get("sender"))}</div>
                <div class="source-label">To</div>
                <div>{safe_text(email.get("recipients"))}</div>
                <div class="source-label">Date</div>
                <div>{safe_text(email.get("date"))}</div>
                <div class="source-label">Email ID</div>
                <div>{safe_text(email.get("email_id"))}</div>
            </div>
            <div class="email-body">{safe_text(email.get("body"), "This email has no body text.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialise persistent search history for the current Streamlit session.
if "history" not in st.session_state:
    st.session_state.history = []
# Track which result cards currently have their source email expanded.
if "open_sources" not in st.session_state:
    st.session_state.open_sources = set()

# Render the Enron Archive branding at the top of the interface.
st.markdown(
    """
    <div class="brand">
        <div class="brand-mark" aria-hidden="true"></div>
        <div class="brand-name">Enron <span>Archive</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
# Show the introductory message only before the user has performed a search.
if not st.session_state.history:
    st.markdown(
        """
        <div class="welcome">
            <h1>Search the correspondence.</h1>
            <p>Ask a question and trace the answer back to its email.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
# Collect a user question and submit it to the archive search pipeline.
with st.form("question_form", clear_on_submit=True, border=False):
    input_column, button_column = st.columns([6.5, 1.5], vertical_alignment="center")

    with input_column:
        question = st.text_input(
            "Question",
            placeholder="Ask about the Enron email archive…",
            label_visibility="collapsed",
        )

    with button_column:
        submitted = st.form_submit_button(
            "Search",
            use_container_width=True,
        )
# Run the complete RAG pipeline when a valid question is submitted.
if submitted and question.strip():
    loading_slot = st.empty()
    loading_slot.markdown(
        loading_animation("Searching the archive and composing an answer…"),
        unsafe_allow_html=True,
    )
    try:
        answer_question = get_answer_question()
        result = answer_question(question.strip())
    except Exception:
        loading_slot.empty()
        st.error(
            "The archive could not complete that search. "
            "Please try the question again."
        )
    else:
        loading_slot.empty()

        # Store the newest result at the top of the session search history.
        st.session_state.history.insert(
            0,
            {
                "id": uuid4().hex,
                "question": question.strip(),
                **result,
            },
        )
# Show the recent-searches heading once the session contains results.
if st.session_state.history:
    st.markdown(
        '<div class="history-title">Recent searches</div>',
        unsafe_allow_html=True,
    )
# Render each question, generated answer, and generation time from the current session.
for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="question-line">
                <span class="question-dot"></span>
                <span>{safe_text(item["question"])}</span>
            </div>
            <div class="answer-copy">{safe_text(item["answer"])}</div>
            <div class="answer-meta">
                Answered in {float(item["generation_seconds"]):.1f} seconds
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Allow each grounded answer to expand or hide its cited original email.
        source_email_id = item.get("source_email_id")

        if source_email_id:
            source_is_open = item["id"] in st.session_state.open_sources
            button_label = "Hide email source" if source_is_open else "View email source"

            if st.button(
                button_label,
                key=f"source_{item['id']}",
                icon="📨",
            ):
                if source_is_open:
                    st.session_state.open_sources.remove(item["id"])
                else:
                    st.session_state.open_sources.add(item["id"])

                st.rerun()
            # Load and display the original cited email when its source panel is opened.
            if source_is_open:
                source_slot = st.empty()
                source_slot.markdown(
                    loading_animation("Opening the original email…"),
                    unsafe_allow_html=True,
                )
                email = find_email(source_email_id)
                source_slot.empty()

                if email is None:
                    st.warning("The source email could not be found in the local archive.")
                else:
                    render_source(email)

