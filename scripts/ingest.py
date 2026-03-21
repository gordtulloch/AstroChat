"""
Document ingestion script for ChromaDB RAG.

Usage:
    python scripts/ingest.py --source data/documents
    python scripts/ingest.py --source path/to/file.txt
    python scripts/ingest.py --source path/to/folder --chunk-size 500 --chunk-overlap 50
    python scripts/ingest.py --clear          # wipe the collection and re-index

Supported file types: .txt, .md, .csv, .pdf, .docx
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

# Allow running from repo root: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_SUPPORTED = {".txt", ".md", ".csv", ".pdf", ".docx"}


def _ocr_pdf_page(path: Path, page_index: int) -> str:
    """Render a single PDF page to an image and return OCR'd text."""
    import io

    import fitz  # pymupdf
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(path))
    page = doc[page_index]
    # 2× zoom gives ~150 dpi → good OCR accuracy without being too slow
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _read_pdf(path: Path, ocr: bool = False) -> str:
    """Extract text from a PDF using pdfplumber, with optional OCR for image-bearing pages."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if ocr and page.images:
                try:
                    ocr_text = _ocr_pdf_page(path, page.page_number - 1).strip()
                    if ocr_text and ocr_text not in text:
                        text = (text + "\n" + ocr_text).strip()
                except Exception as exc:
                    logger.warning(
                        "OCR failed on page %d of %s: %s",
                        page.page_number, path.name, exc,
                    )
            if text:
                pages.append(text)
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs if para.text)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def _stable_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest(
    source: Path,
    retriever: Retriever,
    chunk_size: int,
    chunk_overlap: int,
    ocr: bool = False,
) -> int:
    """Ingest a single file or all supported files in a directory. Returns chunk count."""
    files: list[Path] = []
    if source.is_dir():
        for ext in _SUPPORTED:
            files.extend(source.rglob(f"*{ext}"))
    elif source.suffix in _SUPPORTED:
        files = [source]
    else:
        logger.warning("Unsupported file type: %s — skipping", source)
        return 0

    total = 0
    for file in files:
        try:
            if file.suffix == ".pdf":
                text = _read_pdf(file, ocr=ocr)
            elif file.suffix == ".docx":
                text = _read_docx(file)
            else:
                text = file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Could not read %s: %s", file, exc)
            continue

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            continue

        ids = [_stable_id(str(file), i) for i in range(len(chunks))]
        metadatas = [{"source": str(file), "chunk": i} for i in range(len(chunks))]

        retriever.add_documents(chunks, ids, metadatas)
        logger.info("Ingested %s → %d chunk(s)", file.name, len(chunks))
        total += len(chunks)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--source",
        type=Path,
        default=_REPO_ROOT / "data" / "documents",
        help="File or directory to ingest (default: data/documents/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Characters per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in characters (default: 50)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing documents before ingesting",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="OCR images embedded in PDF pages (requires Tesseract + pymupdf)",
    )
    args = parser.parse_args()

    retriever = Retriever(
        db_path=str(_REPO_ROOT / settings.chroma_db_path),
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        top_k=settings.rag_top_k,
    )
    retriever.start()

    if not retriever.available:
        logger.error("ChromaDB could not be initialised — is chromadb installed?")
        sys.exit(1)

    if args.clear:
        import chromadb
        client = chromadb.PersistentClient(path=str(_REPO_ROOT / settings.chroma_db_path))
        client.delete_collection(settings.chroma_collection)
        retriever.start()  # recreate empty collection
        logger.info("Collection cleared")

    source = args.source.resolve()
    if not source.exists():
        logger.error("Source path does not exist: %s", source)
        sys.exit(1)

    if args.ocr:
        logger.info("OCR enabled — images in PDF pages will be OCR'd")
    count = ingest(source, retriever, args.chunk_size, args.chunk_overlap, ocr=args.ocr)
    logger.info("Done — %d total chunk(s) in collection (total stored: %d)", count, retriever.document_count)


if __name__ == "__main__":
    main()
