"""
summarizer.py — Abstractive summarization using distilbart
Model: sshleifer/distilbart-cnn-12-6

WHY abstractive not extractive:
Extractive: picks existing sentences from document — no understanding needed.
Abstractive: generates NEW sentences summarizing meaning — uses transformer decoder.
distilbart is a distilled version of BART (Bidirectional Auto-Regressive Transformer).
Trained on CNN/DailyMail news articles — generalizes well to reports and papers.

WHY sshleifer/distilbart-cnn-12-6 not facebook/bart-large-cnn:
bart-large-cnn is 400MB+ — times out on Render free tier cold start.
distilbart-cnn-12-6 is ~300MB lighter — works reliably on free deployment.
Performance difference is minimal for our use case.
"""

from transformers import pipeline  # HuggingFace high-level pipeline API

# load summarization pipeline once at module level
# WHY once: same reason as NER — model load is expensive, reuse is cheap
summarizer_model = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)


def summarize_text(text: str) -> str:
    """
    Generate an abstractive summary of the input text.
    Returns a clean summary string.

    max_length=150: maximum tokens in generated summary
    min_length=40:  minimum tokens — prevents trivially short summaries
    do_sample=False: greedy decoding — deterministic output
    WHY do_sample=False: we want consistent summaries, not random variation
    """
    # guard: if text is too short, no need to summarize
    if len(text.split()) < 50:
        return text  # return as-is if already short

    # run summarization model
    # result is a list with one dict: [{"summary_text": "..."}]
    result = summarizer_model(
        text,
        max_length=150,    # max tokens in output summary
        min_length=40,     # min tokens in output summary
        do_sample=False    # deterministic output (no randomness)
    )

    # extract the summary string from result
    summary = result[0]["summary_text"]

    return summary.strip()
