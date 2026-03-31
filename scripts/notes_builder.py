#!/usr/bin/env python3
"""
notes_builder.py

Generates a structured PDF study-notes document for a given quiz config.
Reuses the same RAG pipeline (Cloud SQL + OpenAI embeddings + Groq).

All customisation is expressed in a single CLI command so every flag maps
directly to a future web-UI form field with no interactive prompts needed.

Usage (single-command, all options):
    python scripts/notes_builder.py \
        --topics "Hash Tables,Joins Algorithms,Sorting and Aggregations" \
        --output-name week4_notes \
        --course-name "Database Systems" \
        --question-styles conceptual,scenario,definition \
        --difficulty medium \
        --source-filter "lecture5.pdf" \
        --retrieval-k 6 \
        --max-docs 12

    # Or load topics from an existing quiz config:
    python scripts/notes_builder.py --config my_quiz.json --output-name week4_notes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.cloud import storage

# ── Reuse project internals ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from scripts.quiz_builder import (
    AppConfig,
    load_config,
    retrieve_context,
    call_groq_json,
    format_context_blocks,
    utc_compact_ts,
)

# ── PDF library ───────────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    raise RuntimeError("Run: pip install reportlab")

# ── Constants ─────────────────────────────────────────────────────────────────
GCS_NOTES_PREFIX = "knowledge_base/notes"
OUTPUT_PATH      = Path(os.getenv("OUTPUT_PATH", str(PROJECT_ROOT / "output")))

BRAND_BLUE   = colors.HexColor("#1A3C6E")
BRAND_ACCENT = colors.HexColor("#2E86AB")
LIGHT_GRAY   = colors.HexColor("#F5F5F5")
MID_GRAY     = colors.HexColor("#CCCCCC")

VALID_STYLES      = {"conceptual", "scenario", "definition"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

QUESTION_STYLE_LABELS = {
    "conceptual": "Conceptual Questions",
    "scenario":   "Scenario-Based Questions",
    "definition": "Definition Questions",
}

DIFFICULTY_INSTRUCTIONS = {
    "easy":   "Questions should be straightforward and test basic recall.",
    "medium": "Questions should require some reasoning and understanding.",
    "hard":   "Questions should require deep understanding or multi-step reasoning.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

_LOWERCASE_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor",
    "on", "at", "to", "by", "in", "of", "up", "as", "is",
    "vs", "via",
}


def to_title_case(text: str) -> str:
    if not text:
        return text
    words = text.strip().split()
    result = []
    for i, word in enumerate(words):
        if word.isupper() and len(word) > 1:
            result.append(word)
        elif i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
        elif word.lower() in _LOWERCASE_WORDS:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    return " ".join(result)


def clean_answer_text(text: str) -> str:
    return re.sub(r"^(answer\s*:\s*|a\s*:\s*)", "", text, flags=re.IGNORECASE).strip()


def sanitise_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "notes"


def parse_csv_arg(value: str, valid: set, flag: str) -> List[str]:
    """Parse a comma-separated CLI arg, validate each item, and return a list."""
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad   = [i for i in items if i not in valid]
    if bad:
        print(f"[ERROR] Invalid value(s) for {flag}: {bad}. Choose from: {sorted(valid)}")
        sys.exit(1)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_prompt(topic: str, documents, difficulty: str) -> str:
    ctx  = format_context_blocks(documents)
    diff = DIFFICULTY_INSTRUCTIONS[difficulty]
    return f"""
You are creating study notes for a Database Systems course.
Topic: {topic}
Depth: {diff}

Using ONLY the context below, write a 4-6 sentence paragraph that summarises this topic
clearly for a student preparing for an exam.
Use correct grammar and capitalisation throughout.

Return ONLY a JSON object:
{{"summary": "..."}}

Context:
{ctx}
""".strip()


def build_concepts_prompt(topic: str, documents, difficulty: str) -> str:
    ctx  = format_context_blocks(documents)
    diff = DIFFICULTY_INSTRUCTIONS[difficulty]
    return f"""
You are creating study notes for a Database Systems course.
Topic: {topic}
Depth: {diff}

Extract 4-7 key concepts or terms from the context below.
For each concept provide a concise definition (1-2 sentences).
Use correct capitalisation for technical terms (e.g. "B+ Tree", "DBMS", "SQL").

Return ONLY a JSON object:
{{"concepts": [{{"term": "...", "definition": "..."}}]}}

Context:
{ctx}
""".strip()


def build_practice_qa_prompt(topic: str, documents, style: str, difficulty: str) -> str:
    ctx  = format_context_blocks(documents)
    diff = DIFFICULTY_INSTRUCTIONS[difficulty]
    style_instructions = {
        "conceptual": (
            "Ask 3 questions about underlying concepts or principles. "
            "Each answer should explain the 'why' or 'how'."
        ),
        "scenario": (
            "Present 3 realistic database scenarios and ask the student to apply their knowledge. "
            "Each answer should reason through the scenario step by step."
        ),
        "definition": (
            "Ask the student to define 3 key terms related to this topic. "
            "Each answer should be a clear, precise definition with an example where relevant."
        ),
    }
    return f"""
You are creating study notes for a Database Systems course.
Topic: {topic}
Question style: {style}
Difficulty: {diff}

{style_instructions[style]}

Use correct grammar and capitalisation throughout.
Return ONLY a JSON object:
{{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}

Context:
{ctx}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_topic_notes(
    cfg: AppConfig,
    topic: str,
    retrieval_k: int,
    max_docs: int,
    source_filter: str | None,
    question_styles: List[str],
    difficulty: str,
) -> Dict[str, Any]:
    docs = await retrieve_context(cfg, topic, retrieval_k, max_docs, source_filter)
    if not docs:
        print(f"[WARN] No documents retrieved for topic: {topic}")
        return {"topic": topic, "summary": "", "concepts": [], "qa_by_style": {}}

    print(f"  [→] Summary ...")
    summary_raw = await call_groq_json(cfg, build_summary_prompt(topic, docs, difficulty))
    summary     = summary_raw.get("summary", "") if isinstance(summary_raw, dict) else ""

    print(f"  [→] Key concepts ...")
    concepts_raw = await call_groq_json(cfg, build_concepts_prompt(topic, docs, difficulty))
    concepts     = concepts_raw.get("concepts", []) if isinstance(concepts_raw, dict) else []

    qa_by_style: Dict[str, List[Dict]] = {}
    for style in question_styles:
        print(f"  [→] {style} questions ...")
        qa_raw = await call_groq_json(cfg, build_practice_qa_prompt(topic, docs, style, difficulty))
        pairs  = qa_raw.get("qa_pairs", []) if isinstance(qa_raw, dict) else []
        for p in pairs:
            p["answer"] = clean_answer_text(p.get("answer", ""))
        qa_by_style[style] = pairs

    return {
        "topic":       to_title_case(topic),
        "summary":     summary,
        "concepts":    concepts,
        "qa_by_style": qa_by_style,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"],
            fontSize=28, textColor=BRAND_BLUE,
            spaceAfter=12, alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"],
            fontSize=13, textColor=colors.gray,
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "topic_heading": ParagraphStyle(
            "topic_heading", parent=base["Heading1"],
            fontSize=16, textColor=BRAND_BLUE,
            spaceBefore=18, spaceAfter=8,
        ),
        "section_heading": ParagraphStyle(
            "section_heading", parent=base["Heading2"],
            fontSize=12, textColor=BRAND_ACCENT,
            spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, leading=15,
            textColor=colors.black, spaceAfter=6,
        ),
        "concept_term": ParagraphStyle(
            "concept_term", parent=base["Normal"],
            fontSize=10, leading=14, textColor=BRAND_BLUE,
        ),
        "concept_def": ParagraphStyle(
            "concept_def", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.black,
        ),
        "question": ParagraphStyle(
            "question", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.black,
        ),
        "answer": ParagraphStyle(
            "answer", parent=base["Normal"],
            fontSize=10, leading=14,
            textColor=colors.HexColor("#333333"),
        ),
    }


def _divider() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=6)


def _concept_table(concepts: List[Dict], styles: Dict) -> Table:
    data = [[
        Paragraph("<b>Term</b>", styles["concept_term"]),
        Paragraph("<b>Definition</b>", styles["concept_term"]),
    ]]
    for c in concepts:
        term = to_title_case(c.get("term", ""))
        defn = c.get("definition", "")
        data.append([
            Paragraph(f"<b>{term}</b>", styles["concept_term"]),
            Paragraph(defn, styles["concept_def"]),
        ])
    tbl = Table(data, colWidths=[4.5 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  BRAND_BLUE),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.4, MID_GRAY),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return tbl


def build_pdf(
    notes_data: List[Dict[str, Any]],
    course_name: str    = "Database Systems",
    question_styles: List[str] = None,
    difficulty: str     = "medium",
) -> bytes:
    if question_styles is None:
        question_styles = list(VALID_STYLES)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.2 * cm,  bottomMargin=2.2 * cm,
    )
    styles = _build_styles()
    story  = []

    # Cover
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(course_name, styles["cover_title"]))
    story.append(Paragraph("Study Notes", styles["cover_sub"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(_divider())
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Topics Covered", styles["section_heading"]))
    toc_items = [
        ListItem(Paragraph(nd["topic"], styles["body"]), bulletColor=BRAND_ACCENT)
        for nd in notes_data if nd.get("summary") or nd.get("concepts")
    ]
    if toc_items:
        story.append(ListFlowable(toc_items, bulletType="bullet"))
    story.append(PageBreak())

    # Topic sections
    for nd in notes_data:
        topic       = nd.get("topic", "Unknown Topic")
        summary     = nd.get("summary", "")
        concepts    = nd.get("concepts", [])
        qa_by_style = nd.get("qa_by_style", {})

        if not summary and not concepts and not qa_by_style:
            print(f"[WARN] Skipping empty section for topic: {topic}")
            continue

        story.append(Paragraph(topic, styles["topic_heading"]))
        story.append(_divider())

        if summary:
            story.append(Paragraph("Topic Summary", styles["section_heading"]))
            story.append(Paragraph(summary, styles["body"]))
            story.append(Spacer(1, 0.3 * cm))

        if concepts:
            story.append(Paragraph("Key Concepts & Definitions", styles["section_heading"]))
            story.append(_concept_table(concepts, styles))
            story.append(Spacer(1, 0.4 * cm))

        for style in question_styles:
            pairs = qa_by_style.get(style, [])
            if not pairs:
                continue
            story.append(Paragraph(QUESTION_STYLE_LABELS[style], styles["section_heading"]))
            for i, qa in enumerate(pairs, 1):
                story.append(Paragraph(f"<b>Q{i}.</b> {qa.get('question','')}", styles["question"]))
                story.append(Paragraph(f"<b>Answer:</b> {qa.get('answer','')}", styles["answer"]))
                story.append(Spacer(1, 0.25 * cm))

        story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_pdf_to_gcs(cfg: AppConfig, pdf_bytes: bytes, filename: str) -> str:
    client      = storage.Client(project=cfg.project_id)
    object_path = f"{GCS_NOTES_PREFIX}/{filename}"
    blob        = client.bucket(cfg.gcs_bucket).blob(object_path)
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    uri = f"gs://{cfg.gcs_bucket}/{object_path}"
    print(f"[GCS] PDF uploaded → {uri}")
    return uri


def save_pdf_locally(pdf_bytes: bytes, filename: str) -> Path:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PATH / filename
    out.write_bytes(pdf_bytes)
    print(f"[LOCAL] PDF saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_notes(
    cfg: AppConfig,
    topics: List[str],
    retrieval_k: int        = 6,
    max_docs: int           = 12,
    source_filter: str | None = None,
    course_name: str        = "Database Systems",
    output_name: str | None = None,
    question_styles: List[str] = None,
    difficulty: str         = "medium",
) -> str:
    if question_styles is None:
        question_styles = list(VALID_STYLES)

    notes_data = []
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Generating notes for: {topic}")
        nd = await fetch_topic_notes(
            cfg, topic, retrieval_k, max_docs,
            source_filter, question_styles, difficulty,
        )
        notes_data.append(nd)

    filename  = f"{sanitise_filename(output_name)}.pdf" if output_name else f"notes_{utc_compact_ts()}.pdf"

    print("\n[INFO] Rendering PDF ...")
    pdf_bytes = build_pdf(notes_data, course_name=course_name,
                          question_styles=question_styles, difficulty=difficulty)

    save_pdf_locally(pdf_bytes, filename)
    gcs_uri = save_pdf_to_gcs(cfg, pdf_bytes, filename)
    return gcs_uri


# ─────────────────────────────────────────────────────────────────────────────
# CLI  —  every flag = one future web-UI field
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PDF study notes. All options are single-command flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Web-UI field mapping:
  --topics            →  Topic multi-select / tag input
  --output-name       →  "File name" text input
  --course-name       →  "Course name" text input
  --difficulty        →  Difficulty dropdown  (easy | medium | hard)
  --question-styles   →  Question style checkboxes
  --source-filter     →  "Filter by PDF" text input
  --retrieval-k       →  Advanced: retrieval depth slider
  --max-docs          →  Advanced: max context docs slider
  --config            →  "Load from quiz config" file picker
        """,
    )

    # ── Content ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--topics",
        type=str,
        default=None,
        metavar="TOPIC1,TOPIC2,...",
        help='Comma-separated list of topics, e.g. "Hash Tables,Joins Algorithms"',
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        help="Load topics from an existing quiz JSON config instead of --topics.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        metavar="NAME",
        help='PDF filename without extension, e.g. "week4_joins". Defaults to timestamped name.',
    )
    p.add_argument(
        "--course-name",
        type=str,
        default="Database Systems",
        metavar="NAME",
        help='Course title shown on the PDF cover. Default: "Database Systems".',
    )

    # ── Generation options ────────────────────────────────────────────────────
    p.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=sorted(VALID_DIFFICULTIES),
        metavar="LEVEL",
        help="Question and summary depth: easy | medium | hard. Default: medium.",
    )
    p.add_argument(
        "--question-styles",
        type=str,
        default="conceptual,scenario,definition",
        metavar="STYLE1,STYLE2,...",
        help=(
            "Which practice question styles to include. "
            "Comma-separated from: conceptual, scenario, definition. "
            "Default: all three."
        ),
    )

    # ── Retrieval / filtering ─────────────────────────────────────────────────
    p.add_argument(
        "--source-filter",
        type=str,
        default=None,
        metavar="FILENAME",
        help='Restrict retrieval to a specific PDF, e.g. "lecture5.pdf".',
    )
    p.add_argument(
        "--retrieval-k",
        type=int,
        default=6,
        metavar="K",
        help="Number of chunks to retrieve per query. Default: 6.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=12,
        metavar="N",
        help="Maximum unique context documents after dedup. Default: 12.",
    )

    return p.parse_args()


def main() -> None:
    app_cfg = load_config()
    args    = parse_args()

    # Validate and parse --question-styles
    question_styles = parse_csv_arg(
        args.question_styles, VALID_STYLES, "--question-styles"
    )

    # Resolve topics from --topics or --config
    if args.config:
        with open(args.config) as f:
            quiz_cfg = json.load(f)
        topics = [t["topic"] for t in quiz_cfg.get("topics", [])]
        print(f"[INFO] Loaded {len(topics)} topic(s) from {args.config}")

    elif args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    else:
        print("[ERROR] Provide --topics \"Topic1,Topic2\" or --config FILE.")
        sys.exit(1)

    if not topics:
        print("[ERROR] No topics resolved. Exiting.")
        sys.exit(1)

    # Print resolved config (mirrors what a web UI confirmation screen would show)
    print(f"""
[CONFIG] Topics          : {', '.join(topics)}
[CONFIG] Course name     : {args.course_name}
[CONFIG] Difficulty      : {args.difficulty}
[CONFIG] Question styles : {', '.join(question_styles)}
[CONFIG] Output name     : {args.output_name or '(auto-timestamped)'}
[CONFIG] Source filter   : {args.source_filter or 'all sources'}
[CONFIG] Retrieval k     : {args.retrieval_k}
[CONFIG] Max docs        : {args.max_docs}
""")

    gcs_uri = asyncio.run(
        generate_notes(
            cfg=app_cfg,
            topics=topics,
            retrieval_k=args.retrieval_k,
            max_docs=args.max_docs,
            source_filter=args.source_filter,
            course_name=args.course_name,
            output_name=args.output_name,
            question_styles=question_styles,
            difficulty=args.difficulty,
        )
    )
    print(f"\n  Notes ready at: {gcs_uri}")


if __name__ == "__main__":
    main()