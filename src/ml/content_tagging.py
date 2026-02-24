"""
Concept Tagging System using BERT (Multi-Label)
Trains from local PDF knowledge base -> auto-builds JSON training data (weak supervision)
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

warnings.filterwarnings("ignore")

# Make PyTorch fall back safely if MPS is partially available.
# This does NOT force MPS usage; it prevents certain MPS placeholder crashes.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Force CPU for stability (prevents MPS device mixing issues).
DEVICE = torch.device("cpu")


# -------------------------- PDF TEXT EXTRACTION --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyPDF2.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        A single string containing extracted text across all pages.
    """
    try:
        import PyPDF2
    except ImportError as e:
        raise RuntimeError("PyPDF2 is required. Install with: pip install PyPDF2") from e

    text_chunks: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)

    return "\n".join(text_chunks)


def split_into_candidate_snippets(raw_text: str) -> List[str]:
    """
    Split extracted text into candidate snippets used as training examples.

    Strategy:
      1) Normalize whitespace
      2) Split into paragraphs
      3) If a paragraph is long, split into sentence-like chunks and re-aggregate

    Args:
        raw_text: The full extracted text.

    Returns:
        A list of cleaned snippets.
    """
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    snippets: List[str] = []
    for p in paras:
        if len(p) <= 350:
            snippets.append(p)
            continue

        parts = re.split(r"(?<=[\.\?\!])\s+", p)

        chunk: List[str] = []
        chunk_len = 0
        for s in parts:
            s = s.strip()
            if not s:
                continue
            chunk.append(s)
            chunk_len += len(s)
            if chunk_len >= 220:
                snippets.append(" ".join(chunk).strip())
                chunk, chunk_len = [], 0
        if chunk:
            snippets.append(" ".join(chunk).strip())

    cleaned: List[str] = []
    for s in snippets:
        s = s.strip()
        if len(s) < 35:
            continue
        if re.match(r"^\d+$", s):
            continue
        cleaned.append(s)

    return cleaned


# -------------------------- CONCEPT MATCHING (WEAK SUPERVISION) --------------------------
def normalize_text(s: str) -> str:
    """
    Normalize text to create stable keys for deduplication.
    """
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_concept_patterns(concepts: List[str]) -> Dict[str, List[re.Pattern]]:
    """
    Build regex patterns per concept for matching against snippets.

    Uses:
      - Hand-tuned aliases for frequent DBMS notations (e.g., σ, π, ⋈)
      - Otherwise a literal word-boundary match

    Args:
        concepts: List of all concept tags.

    Returns:
        A dictionary mapping concept -> list of compiled regex patterns.
    """
    patterns: Dict[str, List[re.Pattern]] = {}

    aliases: Dict[str, List[str]] = {
        "Relational Algebra": ["relational algebra"],
        "Selection": [r"\bselection\b", r"\bselect\b", r"σ"],
        "Projection": [r"\bprojection\b", r"π"],
        "Cartesian Product": [r"\bcartesian product\b", r"\bcross join\b", r"×"],
        "Join": [r"\bjoin\b", r"⋈", r"▷◁"],
        "GROUP BY": [r"\bgroup by\b"],
        "ORDER BY": [r"\border by\b"],
        "LIMIT": [r"\blimit\b"],
        "OFFSET": [r"\boffset\b"],
        "Window Functions": [r"\bwindow function\b", r"\bover\s*\("],
        "ROW_NUMBER": [r"\brow_number\b", r"\brow number\b"],
        "RANK": [r"\brank\b"],
        "PARTITION BY": [r"\bpartition by\b"],
        "Nested Queries": [r"\bnested quer", r"\bsubquery\b"],
        "Subqueries": [r"\bsubquery\b"],
        "EXISTS": [r"\bexists\b"],
        "IN": [r"\bin\s*\("],
        "ANY": [r"\bany\b"],
        "ALL": [r"\ball\b"],
        "LATERAL Join": [r"\blateral\b"],
        "CTE": [r"\bcte\b", r"\bwith\b"],
        "Common Table Expressions": [r"\bcommon table expression\b", r"\bwith\b"],
        "Recursive CTE": [r"\brecursive\b", r"\bwith recursive\b"],
        "Slotted Pages": [r"\bslotted page", r"\bslot array\b"],
        "Log-Structured Storage": [r"\blog-structured\b", r"\blog structured\b", r"\bcompaction\b"],
        "Write Amplification": [r"\bwrite amplification\b"],
        "Fragmentation": [r"\bfragmentation\b"],
        "Buffer Pool": [r"\bbuffer pool\b", r"\bbufferpool\b"],
        "Buffer Pool Manager": [r"\bbuffer pool manager\b"],
        "Database Pages": [r"\bdatabase page\b", r"\bpage id\b", r"\bpages\b"],
        "Heap File": [r"\bheap file\b"],
        "Page Directory": [r"\bpage directory\b"],
        "NSM": [r"\bnsm\b", r"\bn-ary storage\b", r"\brow store\b"],
        "DSM": [r"\bdsm\b", r"\bdecomposition storage\b", r"\bcolumn store\b"],
        "Implicit Offsets": [r"\bimplicit offsets\b"],
        "Disk I/O Cost": [r"\bdisk i/o\b", r"\bio cost\b"],
        "Sequential I/O": [r"\bsequential\b.*\bio\b", r"\bsequential io\b"],
        "Random I/O": [r"\brandom\b.*\bio\b", r"\brandom io\b"],
        "Checksum": [r"\bchecksum\b"],
        "NULL": [r"\bnull\b"],
        "Primary Key": [r"\bprimary key\b"],
        "Foreign Key": [r"\bforeign key\b"],
        "Constraints": [r"\bconstraint\b", r"\breferential\b"],
        "Declarative Querying": [r"\bdeclarative\b"],
        "Bags vs Sets": [r"\bbags\b", r"\bsets\b", r"\bduplicates\b"],
        "SQL-92": [r"\bsql-92\b", r"\bsql 92\b"],
    }

    for concept in concepts:
        c = concept.strip()
        patt_list: List[re.Pattern] = []

        if c in aliases:
            for a in aliases[c]:
                patt_list.append(re.compile(a, re.IGNORECASE))
        else:
            escaped = re.escape(c)
            if re.search(r"[A-Za-z0-9]", c):
                patt_list.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
            else:
                patt_list.append(re.compile(escaped, re.IGNORECASE))

        patterns[c] = patt_list

    return patterns


def label_snippet_with_concepts(
    snippet: str,
    concept_patterns: Dict[str, List[re.Pattern]],
    min_hits: int = 1,
) -> List[str]:
    """
    Assign concept tags to a snippet via regex matching.

    Args:
        snippet: Candidate text chunk.
        concept_patterns: Concept regex patterns.
        min_hits: Minimum number of matched concepts required to keep the snippet.

    Returns:
        List of matched concepts (deduplicated). Empty list if below min_hits.
    """
    labels: List[str] = []
    for concept, patt_list in concept_patterns.items():
        for p in patt_list:
            if p.search(snippet):
                labels.append(concept)
                break

    seen = set()
    labels = [x for x in labels if not (x in seen or seen.add(x))]

    if len(labels) < min_hits:
        return []
    return labels


def build_training_json_from_pdfs(
    pdf_paths: List[str],
    concepts: List[str],
    out_json_path: str,
    max_samples: int = 1200,
    min_concepts_per_sample: int = 1,
) -> Tuple[str, int]:
    """
    Create a training JSON from PDF files using weak supervision.

    Pipeline:
      - Extract PDF text
      - Split into snippets
      - Label snippets by regex concept matches
      - Deduplicate snippets
      - Shuffle and cap total samples
      - Write JSON to processed folder

    Args:
        pdf_paths: List of PDF paths.
        concepts: List of concept tags.
        out_json_path: Output JSON path.
        max_samples: Max number of examples to save.
        min_concepts_per_sample: Minimum labels required to keep an example.

    Returns:
        (output_path, number_of_samples_written)
    """
    concept_patterns = build_concept_patterns(concepts)
    all_examples: List[Dict[str, List[str]]] = []

    for pdf in pdf_paths:
        raw = extract_text_from_pdf(pdf)
        snippets = split_into_candidate_snippets(raw)

        for s in snippets:
            labels = label_snippet_with_concepts(s, concept_patterns, min_hits=min_concepts_per_sample)
            if not labels:
                continue
            if len(s) > 700:
                continue
            all_examples.append({"text": s, "concepts": labels})

    uniq: Dict[str, Dict[str, List[str]]] = {}
    for ex in all_examples:
        key = normalize_text(ex["text"])
        if key not in uniq or len(ex["concepts"]) > len(uniq[key]["concepts"]):
            uniq[key] = ex
    all_examples = list(uniq.values())

    rng = np.random.RandomState(42)
    rng.shuffle(all_examples)
    all_examples = all_examples[:max_samples]

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    return out_json_path, len(all_examples)


# -------------------------- MODEL + TRAINING --------------------------
class ConceptDataset(torch.utils.data.Dataset):
    """
    Torch dataset wrapper for tokenized text + multi-hot label vectors.
    """

    def __init__(self, encodings, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        # Tokenizer returns python lists/arrays when return_tensors is not used.
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


class ConceptTagger:
    """
    DistilBERT-based multi-label classifier for concept tagging.
    """

    def __init__(self, concept_list: List[str], model_name: str = "distilbert-base-uncased"):
        self.concept_list = concept_list
        self.model_name = model_name

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([concept_list])

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(concept_list),
            problem_type="multi_label_classification",
        )

        # Force model to CPU immediately.
        self.model.to(DEVICE)

        print(f"Initialized ConceptTagger with {len(concept_list)} concepts")
        print(f"Using model: {model_name}")
        print(f"Device: {DEVICE}")

    def prepare_training_data(self, data_file: str):
        """
        Load training examples from JSON and convert labels to multi-hot vectors.
        """
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item["text"] for item in data]
        concepts = [item["concepts"] for item in data]
        labels = np.array([self.mlb.transform([c])[0] for c in concepts], dtype=np.float32)
        return texts, labels

    def _compute_metrics(self, eval_pred):
        """
        Compute micro-level accuracy across all labels.
        """
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int()
        labels_t = torch.tensor(labels).int()
        micro_acc = (preds == labels_t).float().mean().item()
        return {"micro_accuracy": micro_acc}

    def _make_training_args(
        self,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> TrainingArguments:
        """
        Create TrainingArguments in a version-tolerant way.

        Some transformers versions include Apple-specific flags like use_mps_device.
        This helper tries to disable MPS explicitly when possible.
        """
        base_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

        # Try disabling MPS explicitly (older versions support this argument).
        try:
            return TrainingArguments(**base_kwargs, use_mps_device=False)
        except TypeError:
            return TrainingArguments(**base_kwargs)

    def train(
        self,
        train_file: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ):
        """
        Train the multi-label classifier on the generated training JSON.
        """
        print("\nTraining concept tagger...")
        print(f"Training file: {train_file}")

        texts, labels = self.prepare_training_data(train_file)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        print(f"Train samples: {len(train_texts)}")
        print(f"Val samples: {len(val_texts)}")

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        train_dataset = ConceptDataset(train_encodings, train_labels)
        val_dataset = ConceptDataset(val_encodings, val_labels)

        training_args = self._make_training_args(
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

        print("Starting training...")
        trainer.train()

        # After training, force the model back onto CPU before saving/inference.
        self.model.to(DEVICE)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(f"{output_dir}/concept_list.json", "w", encoding="utf-8") as f:
            json.dump(self.concept_list, f, indent=2, ensure_ascii=False)

        print(f"Model saved to {output_dir}")

    def predict(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Predict concept tags for a given input text.

        This forces the model and inputs onto CPU to avoid MPS placeholder errors.
        """
        # Force CPU before inference (Trainer/Accelerate can move the model during training).
        self.model.to(DEVICE)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0].detach().cpu()

        predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0]
        predicted_concepts = [self.concept_list[i] for i in predicted_indices]

        all_probs = {self.concept_list[i]: float(probs[i]) for i in range(len(self.concept_list))}
        sorted_concepts = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        return {
            "predicted_concepts": predicted_concepts,
            "top_5_concepts": [c for c, _ in sorted_concepts[:5]],
            "all_probabilities": all_probs,
            "confidence": float(probs[predicted_indices].mean()) if len(predicted_indices) else 0.0,
        }


# ==================== MAIN ====================
if __name__ == "__main__":

    CONCEPTS = [
        # Relational Model
        "Relational Model", "Relations", "Tuples", "Attributes",
        "Primary Key", "Foreign Key", "Constraints", "Referential Integrity", "NULL", "Schema",

        # Relational Algebra
        "Relational Algebra", "Selection", "Projection", "Union", "Intersection",
        "Difference", "Cartesian Product", "Join", "Declarative Querying", "Query Optimization",

        # SQL
        "SQL", "SQL-92", "DML", "DDL", "DCL", "Bags vs Sets",
        "Aggregate Functions", "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT",
        "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET",
        "String Operations", "LIKE",
        "Window Functions", "ROW_NUMBER", "RANK", "PARTITION BY",
        "Nested Queries", "Subqueries", "EXISTS", "IN", "ANY", "ALL",
        "LATERAL Join", "Common Table Expressions", "CTE", "Recursive CTE",

        # Storage / Buffer / Pages
        "Disk-Oriented DBMS", "Volatile Storage", "Non-Volatile Storage", "Persistent Memory",
        "Storage Hierarchy", "Buffer Pool", "Buffer Pool Manager", "Execution Engine",
        "Database Pages", "Page ID", "Hardware Page", "OS Page",
        "Heap File", "Linked List Heap", "Page Directory", "Fixed-Size Pages",
        "Page Header", "Checksum",

        # Page/Tuple layout
        "Slotted Pages", "Slot Array", "Log-Structured Storage",
        "Fragmentation", "Write Amplification",
        "Tuple Layout", "Tuple Header", "NULL Bitmap", "RID",

        # Storage models / I/O
        "NSM", "DSM", "Implicit Offsets", "Column Store", "Row Store",
        "Sequential I/O", "Random I/O", "Disk I/O Cost", "Page Reads",
        "Worst-Case I/O", "Scan Cost", "Column Pruning", "Tuple Reconstruction",

        # Original internals topics (for future expansion)
        "PAX", "Compression",
        "B+Trees", "Hash Tables", "Linear Probing", "Cuckoo Hashing",
        "Extendible Hashing", "Bloom Filters", "Skip Lists", "Tries",
        "Sorting Algorithms", "External Merge Sort", "Join Algorithms",
        "Nested Loop Join", "Hash Join", "Sort-Merge Join",
        "Cost Models", "Selectivity",
        "ACID", "Serializability", "Two-Phase Locking", "2PL",
        "MVCC", "Timestamp Ordering", "OCC", "Isolation Levels",
        "Deadlock Detection", "Deadlock Prevention",
        "Write-Ahead Logging", "WAL", "ARIES", "Checkpointing",
        "Shadow Paging", "Undo", "Redo",
        "Partitioning", "Replication", "Two-Phase Commit", "2PC",
        "Paxos", "CAP Theorem", "Consensus", "Distributed Transactions",
    ]

    # Build paths relative to this file so it works regardless of current working directory.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
    KB_BASE_DIR = PROJECT_ROOT / "src" / "database" / "data" / "knowledge_base"
    KB_RAW_DIR = KB_BASE_DIR / "raw"
    KB_PROCESSED_DIR = KB_BASE_DIR / "processed"

    pdf_paths = [
        str(KB_RAW_DIR / "lectures.pdf"),
        str(KB_RAW_DIR / "homework_questions_merged.pdf"),
        str(KB_RAW_DIR / "homework_solutions_merged.pdf"),
    ]

    print("Knowledge base files:")
    for p in pdf_paths:
        print(f"- {p}")

    missing = [p for p in pdf_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing knowledge base files:\n" + "\n".join(missing))

    training_json_path = str(KB_PROCESSED_DIR / "concept_training_from_pdfs.json")

    training_json_path, n_samples = build_training_json_from_pdfs(
        pdf_paths=pdf_paths,
        concepts=CONCEPTS,
        out_json_path=training_json_path,
        max_samples=1200,
        min_concepts_per_sample=1,
    )

    print(f"\nTraining data created: {training_json_path}")
    print(f"Samples: {n_samples}")

    tagger = ConceptTagger(CONCEPTS)

    model_out_dir = str(PROJECT_ROOT / "models" / "concept_tagger_dbms_kb")
    tagger.train(
        train_file=training_json_path,
        output_dir=model_out_dir,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
    )

    test_text = "How does external merge sort work with limited buffer pool and why is it needed?"
    prediction = tagger.predict(test_text)

    print("\nTest Prediction:")
    print(f"Text: {test_text}")
    print(f"Predicted: {prediction['predicted_concepts']}")
    print(f"Top 5: {prediction['top_5_concepts']}")