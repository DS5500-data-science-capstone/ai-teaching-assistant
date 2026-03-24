#!/usr/bin/env python3
"""
build_concepts.py

Concept list builder for the capstone AI teaching assistant.

Key requirements:
- Fetch PDFs from GCS (no local persistence)
- Extract candidate concepts without hardcoding a concept list
- Optional syllabus support (but we validate its existence if provided)
- Provide utilities used by content_tagging_2.py

Exports (used by your tagger):
- get_gcs_client
- parse_gs_uri
- list_pdf_blobs
- download_blob_bytes
- concepts_to_json_bytes
- debug_validate_gcs_inputs
- build_concepts_from_gcs
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# GCS utilities
# -----------------------------

def get_gcs_client(service_key_path: Optional[str] = None):
    """
    Create a GCS client.
    If service_key_path is provided, use it; otherwise rely on ADC.
    """
    from google.cloud import storage
    if service_key_path:
        return storage.Client.from_service_account_json(service_key_path)
    return storage.Client()


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    """
    Parse gs://bucket/path -> (bucket, path)
    """
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid gs uri (must start with gs://): {gs_uri}")
    rest = gs_uri[len("gs://") :]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    blob = parts[1].strip() if len(parts) == 2 else ""
    if not bucket:
        raise ValueError(f"Invalid gs uri (empty bucket): {gs_uri}")
    return bucket, blob


def list_pdf_blobs(client, *, bucket: str, prefix: str, max_pdfs: int) -> List[str]:
    """
    Return blob names ending with .pdf under gs://bucket/prefix
    """
    bkt = client.bucket(bucket)
    out: List[str] = []
    for blob in bkt.list_blobs(prefix=prefix):
        if blob.name.lower().endswith(".pdf"):
            out.append(blob.name)
            if len(out) >= max_pdfs:
                break
    return out


def download_blob_bytes(client, *, bucket: str, blob_name: str) -> bytes:
    bkt = client.bucket(bucket)
    blob = bkt.blob(blob_name)
    return blob.download_as_bytes()


def _gcs_blob_exists(client, bucket: str, blob_name: str) -> bool:
    bkt = client.bucket(bucket)
    blob = bkt.blob(blob_name)
    return blob.exists()


def _list_blobs_sample(client, *, bucket: str, prefix: str, max_items: int = 30) -> List[str]:
    bkt = client.bucket(bucket)
    out: List[str] = []
    for blob in bkt.list_blobs(prefix=prefix):
        out.append(blob.name)
        if len(out) >= max_items:
            break
    return out


def debug_validate_gcs_inputs(
    client,
    *,
    pdf_bucket: str,
    pdf_prefix: str,
    syllabus_gs_uri: Optional[str],
) -> None:
    """
    Hard validation (fail-fast) to expose the real root cause:
    - prefix contains objects and at least one PDF
    - syllabus blob exists if provided
    """
    sample = _list_blobs_sample(client, bucket=pdf_bucket, prefix=pdf_prefix, max_items=25)
    if not sample:
        raise ValueError(
            f"[GCS VALIDATION] No objects found under gs://{pdf_bucket}/{pdf_prefix}\n"
            f"Fix your prefix to an existing folder."
        )

    pdfs = [x for x in sample if x.lower().endswith(".pdf")]
    if not pdfs:
        raise ValueError(
            f"[GCS VALIDATION] Objects exist under gs://{pdf_bucket}/{pdf_prefix} but no PDFs found in sample.\n"
            f"Sample:\n" + "\n".join([f"  - {x}" for x in sample])
        )

    if syllabus_gs_uri:
        sbkt, sblob = parse_gs_uri(syllabus_gs_uri)
        if not _gcs_blob_exists(client, sbkt, sblob):
            parent = "/".join(sblob.split("/")[:-1])
            near = _list_blobs_sample(client, bucket=sbkt, prefix=parent, max_items=50) if parent else []
            raise ValueError(
                "[GCS VALIDATION] Syllabus blob NOT FOUND.\n"
                f"Requested: {syllabus_gs_uri}\n"
                f"Bucket: {sbkt}\n"
                f"Blob: {sblob}\n"
                f"Nearby objects (sample):\n"
                + ("\n".join([f"  - gs://{sbkt}/{x}" for x in near]) if near else "  (no nearby objects found)")
                + "\n\nFix concept_syllabus_gs_uri to the correct path, or upload the file there."
            )


def _read_text_gs_uri(client, gs_uri: str) -> str:
    bkt, blob = parse_gs_uri(gs_uri)
    data = download_blob_bytes(client, bucket=bkt, blob_name=blob)
    return data.decode("utf-8", errors="ignore")


# -----------------------------
# PDF text extraction
# -----------------------------

def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: Optional[int]) -> List[str]:
    """
    Returns per-page text list.
    """
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages: List[str] = []
    total = len(reader.pages)
    limit = min(total, max_pages) if max_pages else total
    for i in range(limit):
        t = reader.pages[i].extract_text() or ""
        pages.append(t.replace("\r", "\n"))
    return pages


# -----------------------------
# Concept mining
# -----------------------------

GENERIC_BAD = {
    "introduction",
    "overview",
    "summary",
    "references",
    "appendix",
    "review questions",
    "exercises",
    "problems",
    "bibliography",
    "chapter",
    "part",
    "contents",
    "table of contents",
    "index",
    "example",
    "examples",
    "definition",
    "definitions",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-+]*")
HEADING_LIKE = re.compile(r"^[A-Z][A-Za-z0-9 ,:+\-()/]{3,100}$")
TOC_LINE = re.compile(r"^(.+?)\s+(\d{1,4})\s*$")
ACRONYM_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z\- ]{5,90}?)\s*\(\s*([A-Z]{2,10})\s*\)")


def _normalize(s: str) -> str:
    s = s.strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;—–-")


def _looks_like_concept(s: str) -> bool:
    t = s.lower().strip()
    if len(t) < 3 or len(t) > 110:
        return False
    if t in GENERIC_BAD:
        return False
    if re.fullmatch(r"[\d\W]+", s):
        return False
    if t.count(" ") > 10:
        return False
    if not WORD_RE.search(s):
        return False
    if t.count(".") >= 2:
        return False
    return True


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:80] if s else "concept"


def _mine_structural_candidates(page_text: str) -> List[str]:
    cands: List[str] = []
    for line in page_text.split("\n"):
        line = _normalize(line)
        if not line:
            continue

        m = TOC_LINE.match(line)
        if m:
            title = _normalize(m.group(1))
            if _looks_like_concept(title):
                cands.append(title)
            continue

        if HEADING_LIKE.match(line) and _looks_like_concept(line):
            if not re.match(r"^(chapter|part)\b", line.lower()):
                cands.append(line)
    return cands


def _extract_acronym_pairs(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for m in ACRONYM_PATTERN.finditer(text):
        expanded = _normalize(m.group(1))
        acro = _normalize(m.group(2))
        if 2 <= len(acro) <= 10 and _looks_like_concept(expanded):
            pairs.append((expanded, acro))
    return pairs


def _tfidf_keyphrases(pages: List[str], *, top_k: int) -> Dict[str, float]:
    docs = [re.sub(r"\s+", " ", t) for t in pages if t.strip()]
    if not docs:
        return {}

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.85,
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1

    scored = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_k]
    out: Dict[str, float] = {}
    for phrase, score in scored:
        ph = _normalize(phrase)
        if ph.lower() in GENERIC_BAD:
            continue
        if ph.count(" ") == 0 and len(ph) < 5:
            continue
        if _looks_like_concept(ph):
            out[ph] = float(score)
    return out


def _cluster_terms(terms: List[str], threshold: int = 93) -> List[List[str]]:
    terms = sorted(terms, key=lambda x: (-len(x), x.lower()))
    clusters: List[List[str]] = []
    used: Set[str] = set()

    for i, seed in enumerate(terms):
        if seed in used:
            continue
        cluster = [seed]
        used.add(seed)
        for t in terms[i + 1 :]:
            if t in used:
                continue
            if fuzz.ratio(seed.lower(), t.lower()) >= threshold:
                cluster.append(t)
                used.add(t)
        clusters.append(cluster)
    return clusters


def _pick_canonical(cluster: List[str]) -> str:
    def score(t: str) -> Tuple[int, int, int]:
        allcaps_penalty = -10 if (t.isupper() and len(t) > 4) else 0
        alpha = sum(ch.isalpha() for ch in t)
        return (len(t) + allcaps_penalty, alpha, -abs(t.count(" ") - 2))
    return sorted(cluster, key=score, reverse=True)[0]


def _build_sources_map(book_pages: Dict[str, List[str]], terms: List[str]) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = {}
    for book, pages in book_pages.items():
        out[book] = {}
        lower_pages = [p.lower() for p in pages]
        for term in terms:
            tt = term.lower()
            hits: List[int] = []
            for i, page_text in enumerate(lower_pages, start=1):
                if tt in page_text:
                    hits.append(i)
            if hits:
                out[book][term] = hits
    return out


@dataclass
class Concept:
    concept_id: str
    canonical_name: str
    aliases: List[str]
    sources: List[Dict[str, Any]]
    signals: Dict[str, Any]
    definition_snippet: Optional[str] = None


def concepts_to_json_bytes(concepts: List[Dict[str, Any]]) -> bytes:
    return json.dumps(concepts, indent=2, ensure_ascii=False).encode("utf-8")


def build_concepts_from_gcs(
    *,
    service_key_path: Optional[str],
    bucket: str,
    pdf_prefix: str,
    max_pdfs: int,
    max_pages_per_pdf: Optional[int],
    top_k: int,
    final_k: int,
    syllabus_gs_uri: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Build concept objects from PDFs stored in GCS.
    Returns list[dict] ready to be used by content_tagging_2.py.
    """
    client = get_gcs_client(service_key_path)

    # Root-cause validation: if syllabus doesn't exist, fail with a diagnostic listing.
    debug_validate_gcs_inputs(client, pdf_bucket=bucket, pdf_prefix=pdf_prefix, syllabus_gs_uri=syllabus_gs_uri)

    syllabus_text: Optional[str] = _read_text_gs_uri(client, syllabus_gs_uri) if syllabus_gs_uri else None

    blob_names = list_pdf_blobs(client, bucket=bucket, prefix=pdf_prefix, max_pdfs=max_pdfs)
    if not blob_names:
        raise ValueError(f"No PDFs found in gs://{bucket}/{pdf_prefix}")

    # Read PDFs into per-book page text
    book_pages: Dict[str, List[str]] = {}
    all_pages_flat: List[str] = []
    structural: Set[str] = set()
    acronym_pairs: List[Tuple[str, str]] = []

    for blob_name in blob_names:
        pdf_bytes = download_blob_bytes(client, bucket=bucket, blob_name=blob_name)
        pages = _extract_text_from_pdf_bytes(pdf_bytes, max_pages=max_pages_per_pdf)
        book_key = blob_name.split("/")[-1].rsplit(".", 1)[0]  # filename stem
        book_pages[book_key] = pages
        all_pages_flat.extend(pages)

        for page in pages:
            structural.update(_mine_structural_candidates(page))
            acronym_pairs.extend(_extract_acronym_pairs(page))

    tfidf_ph = _tfidf_keyphrases(all_pages_flat, top_k=top_k)

    candidates: Set[str] = set(structural) | set(tfidf_ph.keys())
    for expanded, acro in acronym_pairs:
        candidates.add(expanded)
        candidates.add(acro)

    if syllabus_text:
        # add a small amount of candidate mining from syllabus
        for line in syllabus_text.splitlines():
            line = _normalize(line)
            if _looks_like_concept(line):
                candidates.add(line)

    candidates = {_normalize(t) for t in candidates if _looks_like_concept(t)}
    clusters = _cluster_terms(sorted(candidates), threshold=93)

    flat_terms = sorted({t for cl in clusters for t in cl})
    sources_map = _build_sources_map(book_pages, flat_terms)

    concepts: List[Concept] = []
    for cl in clusters:
        canonical = _pick_canonical(cl)
        aliases = sorted({t for t in cl if t != canonical})

        sources: List[Dict[str, Any]] = []
        for book in sources_map:
            pages_hit: Set[int] = set()
            for t in [canonical] + aliases:
                pages_hit.update(sources_map[book].get(t, []))
            if pages_hit:
                sources.append({"book": book, "pages": sorted(pages_hit)[:80]})

        if not sources:
            continue

        concepts.append(
            Concept(
                concept_id=_slugify(canonical),
                canonical_name=canonical,
                aliases=aliases[:50],
                sources=sources,
                signals={
                    "tfidf_score": float(tfidf_ph.get(canonical, 0.0)),
                    "cluster_size": len(cl),
                },
                definition_snippet=None,
            )
        )

    # de-dupe concept_id
    seen: Dict[str, int] = {}
    for c in concepts:
        if c.concept_id in seen:
            seen[c.concept_id] += 1
            c.concept_id = f"{c.concept_id}-{seen[c.concept_id]}"
        else:
            seen[c.concept_id] = 1

    # sort by tfidf_score primarily
    concepts_sorted = sorted(concepts, key=lambda x: (x.signals.get("tfidf_score", 0.0), len(x.aliases)), reverse=True)
    concepts_sorted = concepts_sorted[: final_k]

    return [asdict(c) for c in concepts_sorted]