# Local AI Structify

Local AI Structify is an offline, CPU-first document intelligence application for extracting text and structured data from local files. It combines file ingestion, OCR, transcription, preprocessing, local AI inference, schema mapping, validation, search, and exports in a Streamlit UI.

## Features
- Offline-first processing with local storage and local model support.
- Streamlit interface for upload, results, search, dashboard, history, and settings.
- File validation, deduplication, and ingestion management.
- PDF, image, text, data, audio, and video processing workflows.
- OCR with Tesseract and PDF image conversion with Poppler.
- Audio and video transcription with faster-whisper and FFmpeg.
- Local AI schema extraction through Ollama by default.
- Text cleaning, chunking, metadata enrichment, and feature extraction.
- SQLite-backed document storage and search services.
- Export support for JSON, CSV, Excel, text reports, and ZIP bundles.

## Supported File Types
- Documents: `.pdf`, `.txt`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`
- Audio: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Data: `.csv`, `.json`

## Requirements
- Python 3.12 or newer
- Tesseract OCR
- Poppler utilities, including `pdftoppm`
- FFmpeg
- Ollama, for local LLM inference

Python dependencies are listed in `requirements.txt`.

Optional Python dependencies:
- `llama-cpp-python` for llama.cpp inference support
- `openpyxl` for Excel export support
- `psutil` for enhanced system monitoring

## Setup
1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Create a local environment file if you need overrides.

```bash
cp .env.example .env
```

4. Install and start external tools as needed.

```bash
ollama serve
ollama pull llama3.2:latest
```

Make sure `tesseract`, `pdftoppm`, `ffmpeg`, and `ollama` are available on your `PATH`.

## Run
Validate the environment only:

```bash
python app.py --check
```

Launch the Streamlit UI:

```bash
python app.py
```

The app runs Streamlit with `ui/app.py` and opens a local interface for uploading, processing, searching, and exporting documents.

## Configuration
Configuration is loaded from environment variables or a `.env` file. Defaults are defined in `config.py`, and examples are provided in `.env.example`.

Common settings:
- `DATABASE_URL` controls the SQLite database location.
- `MAX_UPLOAD_SIZE_MB` controls upload limits.
- `OCR_LANGUAGE`, `OCR_DPI`, and `TESSERACT_CMD` control OCR behavior.
- `WHISPER_MODEL_DIR`, `WHISPER_DEVICE`, and `WHISPER_THREADS` control transcription.
- `LLM_PROVIDER`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL` control local AI inference.
- `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, and `CACHE_MAX_SIZE_MB` control caching.

Runtime directories such as `uploads/`, `cache/`, `data/`, `models/`, and `logs/` are created automatically.

## Project Structure
- `app.py` - environment check and Streamlit launcher
- `ui/` - Streamlit application pages and components
- `ingestion/` - upload validation, storage, and deduplication
- `extraction/` - PDF, OCR, audio, and video extraction
- `preprocessing/` - text cleanup, metadata, features, and chunking
- `ai/` - local model provider and inference helpers
- `schema_mapping/` - schema extraction, validation, and repair
- `database/` - SQLite models, sessions, repository, and search
- `services/` - pipeline, document, search, cache, export, monitoring, and recovery services
- `tests/` - automated tests
- `docs/` - project process, growth, and expansion documentation

## Testing
Run the test suite with:

```bash
pytest
```

## Documentation
- `docs/FEEDBACK_PROCESS.md` explains how feedback is collected and converted into improvements.
- `docs/GROWTH_PLAN.md` outlines adoption and community growth goals.
- `docs/EXPANSION_PLAN.md` outlines institutional and geographical expansion goals.

## Privacy
Local AI Structify is designed for offline document processing. Uploaded files, extracted text, structured data, cache files, logs, and exports remain on the local machine unless the user explicitly moves or shares them.
