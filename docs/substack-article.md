# Building AstroChat: A Private, Local AI Assistant for Schools

*How we built a fully self-hosted AI chat application — no cloud subscriptions, no data leaving the building.*

---

If you work in a school division and have been watching the AI space, you've probably felt the tension. On one hand, tools like ChatGPT are genuinely useful for staff. On the other hand, the idea of school data — staff queries, internal documents, operational knowledge — flowing through a third-party's servers is uncomfortable at best and a policy violation at worst.

AstroChat is our answer to that tension. It's a self-hosted AI assistant built entirely on open-source tools that runs on our own hardware. Here's how it works, what it can do, and how we recently made it smarter about understanding our own documents.

---

## What is AstroChat?

AstroChat is a web-based chat application — think ChatGPT, but running entirely on a server you control, with no data ever leaving your network.

Staff open a browser, type a question, and get a streamed, conversational response from a local AI model. They can save and revisit past conversations, adjust how the AI responds (more creative vs. more precise), and — crucially — get answers that are grounded in **our own documents**.

That last part is where things get interesting.

---

## The Core Technology Stack

Before diving into features, a quick tour of the building blocks:

**The AI model** is [Mistral 7B Instruct](https://mistral.ai/), a powerful open-source language model small enough to run on a single server with a decent GPU. We use a compressed version (Q3_K_M quantization) that trades a small amount of accuracy for substantially less memory and faster responses. It runs via **llama.cpp** — a high-performance inference engine that makes running large language models on commodity hardware practical.

**The backend** is a [FastAPI](https://fastapi.tiangolo.com/) Python application. FastAPI is modern, fast, and makes it easy to stream responses to the browser in real time — so you see the AI's answer appearing word by word rather than waiting for the whole thing to generate.

**The frontend** is a clean, single-page web app written in plain JavaScript — no heavy frameworks, no build pipeline. It just works in any browser.

---

## The Key Feature: RAG (Retrieval-Augmented Generation)

A vanilla AI model only knows what it was trained on. It doesn't know our policies, our procedures, our local history, or anything updated after its training cutoff. RAG solves this.

Here's the idea in plain terms:

1. We feed the AI a library of our own documents — policy manuals, historical records, meeting notes, web pages from our public site.
2. Those documents are broken into chunks and converted into mathematical "embeddings" — a way of capturing the meaning of text as numbers — and stored in a **vector database** (we use [ChromaDB](https://www.trychroma.com/)).
3. When a staff member asks a question, we first search the vector database for the most relevant document chunks based on the meaning of the question (not just keyword matching).
4. Those chunks are silently injected into the AI's context before it answers.

The result: the AI can say "According to the 2024 policy manual, the procedure for X is..." — and mean it, because it's actually reading the relevant excerpt.

---

## Tools: The AI That Can Take Actions

AstroChat doesn't just talk — it can also **do things**.

We've built a connection to an external tool server using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), an emerging standard for AI tool use. When the AI decides it needs to look something up or perform an action, it doesn't just guess; it calls the appropriate tool, gets back a result, and incorporates that into its response.

The tool loop looks like this:

```
User asks a question
  → AI decides it needs a tool
  → AI calls the tool (e.g. "search the directory for staff member X")
  → Tool returns a result
  → AI reads the result and continues its response
  → Final answer delivered to the user
```

The user sees the tool calls transparently — expandable panels in the chat bubble show exactly which tool was called, with what arguments, and what it returned. No black boxes.

---

## Privacy and Authentication

Because this is a school environment, we built in optional **Microsoft Entra ID (Azure AD) authentication**. When enabled, staff sign in with their existing school credentials via a pop-up login. Their name is automatically recognized and woven into the AI's context, so responses can be personalized.

All API endpoints require a valid, cryptographically verified token — we validate the JWT signature against Microsoft's public keys, checking issuer and audience claims. When authentication is off (during development or in low-risk deployments), everything still works normally.

---

## Making AstroChat Smarter About Our Documents

The original version of AstroChat could only ingest plain text files (`.txt`, `.md`, `.csv`). Most of our real-world documents — policy manuals, historical records, reports — live in PDF or Word format. We've just shipped a major upgrade to the document ingestion pipeline to address this.

### What's New: PDF and DOCX Support

The ingestion script (`ingest.py`) that feeds documents into the knowledge base now supports:

- **PDF files** — using [pdfplumber](https://github.com/jsvine/pdfplumber), which extracts text from PDFs cleanly, handling multi-column layouts and preserving reading order.
- **DOCX files** — using [python-docx](https://python-docx.readthedocs.io/), which reads Microsoft Word documents paragraph by paragraph.

Both file types are detected automatically. You can drop a folder of mixed files — some `.txt`, some `.pdf`, some `.docx` — and run the ingestion script once:

```bash
python scripts/ingest.py --source data/documents
```

It handles everything.

### The Harder Problem: Images Inside PDFs

PDFs come in two flavours. **Text PDFs** — like those exported from Word or Google Docs — contain actual text characters that software can read directly. **Scanned PDFs** — like a document that was photocopied and scanned — contain only images of text. A standard text extractor reads these as blank pages.

A huge portion of historical school documents fall into the second category.

We solved this with **OCR** (Optical Character Recognition) — the technology that converts images of text back into actual, searchable, readable text.

Here's how our implementation works:

1. For each page in a PDF, we run the normal text extraction first.
2. If that page contains embedded images **and** the `--ocr` flag is passed, we additionally render the page to a high-resolution image using [PyMuPDF](https://pymupdf.readthedocs.io/) (at 2× zoom for accuracy) and pass it through [Tesseract](https://tesseract-ocr.github.io/) — the gold-standard open-source OCR engine.
3. The OCR output is merged with the direct text extraction, avoiding duplicates.

The result: a scanned policy document from 1987 can now be read, chunked, embedded, and retrieved just like any modern document.

To use OCR:

```bash
python scripts/ingest.py --source data/documents --ocr
```

OCR is opt-in — it adds processing time, and most modern PDFs don't need it. But for historical archives or scanned records, it's a game-changer.

### Under the Hood: The Chunking Process

Regardless of file type, all ingested text goes through the same pipeline:

1. **Extract** — get the raw text from the file
2. **Chunk** — split it into overlapping windows (default: 500 characters with a 50-character overlap). Overlapping chunks prevent information from falling through the cracks at chunk boundaries.
3. **Embed** — convert each chunk to a vector using the `all-MiniLM-L6-v2` sentence-transformers model (a small, fast, surprisingly capable embedding model)
4. **Store** — write chunks and their embeddings to ChromaDB with stable, deterministic IDs so re-running the ingestion doesn't create duplicates

Chunk size and overlap are configurable:

```bash
python scripts/ingest.py --source data/documents --chunk-size 800 --chunk-overlap 100
```

---

## Web Ingestion: Crawling Our Own Website

In addition to local files, AstroChat can crawl a website and ingest its content. Our `web_ingest.py` script uses [Scrapy](https://scrapy.org/) — a production-grade web crawling framework — to politely traverse a site, respecting `robots.txt` and rate limits.

It ingests three content types found on the web:

- **HTML pages** — cleaned of navigation, footers, and scripts; only the meaningful body content is kept
- **Linked PDFs** — extracted using pdfplumber
- **Linked DOCX files** — extracted using python-docx

This means staff can ask questions about content on our public website — announcements, program descriptions, contact information — without anyone manually copying text.

```bash
python scripts/web_ingest.py --url https://www.sunrisesd.ca --depth 3
```

---

## The Full Architecture at a Glance

```
Browser
  │
  ▼
FastAPI App (localhost:8080)
  ├── RAG Pipeline
  │     └── ChromaDB vector store  ←── ingest.py / web_ingest.py
  │           └── Document chunks from .txt, .md, .pdf, .docx, web pages
  │
  ├── LLM Client
  │     └── Mistral 7B via llama.cpp (localhost:8081)
  │
  └── MCP Tool Client
        └── Remote tool server (school division infrastructure)
```

When a question comes in:
1. Relevant document chunks are retrieved from ChromaDB
2. Those chunks are added to the AI's context
3. The AI generates a response, potentially calling tools along the way
4. The response streams back to the browser in real time

---

## Why This Matters

The narrative that AI requires cloud services and third-party subscriptions is simply not true anymore. With open-source models, open-source tooling, and a reasonable server, a small organization can run a capable, context-aware AI assistant entirely on infrastructure it controls.

For us, the benefits are clear:
- **Privacy** — staff queries never leave the network
- **Relevance** — the AI knows our documents, our policies, our history
- **Cost** — no per-token API billing; the hardware cost is fixed
- **Control** — we decide what the AI knows, how it behaves, and who can access it

The recent document ingestion enhancements — PDF, DOCX, and OCR support — close the loop on the "knowledge base" side of the equation. The AI is only as useful as the documents it can learn from. Now it can learn from almost anything we have.

---

*AstroChat is built on [FastAPI](https://fastapi.tiangolo.com/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [ChromaDB](https://www.trychroma.com/), [pdfplumber](https://github.com/jsvine/pdfplumber), [PyMuPDF](https://pymupdf.readthedocs.io/), [Tesseract OCR](https://tesseract-ocr.github.io/), and the [Model Context Protocol](https://modelcontextprotocol.io/).*
