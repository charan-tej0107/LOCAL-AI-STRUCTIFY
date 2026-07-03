# Local AI Structify

Local AI Structify is an offline, CPU-first document intelligence application for extracting text and structured data from local files. It combines file ingestion, validation, OCR, transcription, preprocessing, local AI inference, schema mapping, search, exports, monitoring, and recovery in a Streamlit interface.

The project is designed for students, educators, researchers, and organizations that need privacy-focused document processing without sending files to cloud services.

## What This Project Does
- Uploads and validates local documents, images, audio, video, text, CSV, and JSON files.
- Deduplicates uploaded files using file hashes.
- Extracts text from PDFs, images, audio, video, and plain text/data files.
- Uses Tesseract OCR for image and scanned PDF extraction.
- Uses faster-whisper for local audio transcription.
- Uses FFmpeg for video audio extraction.
- Cleans, chunks, and enriches extracted text with metadata and features.
- Uses a local AI provider, Ollama by default, for structured schema extraction.
- Stores document metadata, extracted text, structured JSON, confidence scores, and processing status locally.
- Provides search, history, dashboard, results, settings, and export workflows through Streamlit.
- Exports processed data as JSON, CSV, Excel, text reports, or ZIP bundles.

## Supported File Types
- Documents: `.pdf`, `.txt`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`
- Audio: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Data: `.csv`, `.json`

## System Requirements
- Python 3.12 or newer
- Tesseract OCR
- Poppler utilities, including `pdftoppm`
- FFmpeg
- Ollama for local LLM inference
- At least 1 GB free disk space for local data, cache, logs, models, and uploads

The application health check validates Python, Tesseract, Poppler, FFmpeg, Ollama, disk space, and required Python packages before launching the UI.

## Fresh Setup From Clone
Follow these steps to replicate the project on a new machine.

### 1. Clone The Repository
```bash
git clone <repository-url>
cd local-ai-structify
```

### 2. Check Python
```bash
python --version
```

Use Python `3.12` or newer. If your system command points to an older version, use the correct executable, for example `python3.12`.

### 3. Create A Virtual Environment
Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Python Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional development and export dependencies:

```bash
pip install pytest openpyxl psutil
```

Optional llama.cpp backend support:

```bash
pip install llama-cpp-python
```

## Install External Tools
The project depends on several command-line tools. They must be installed and available on your `PATH`.

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y tesseract-ocr poppler-utils ffmpeg
```

Install Ollama using the official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
Using Homebrew:

```bash
brew install tesseract poppler ffmpeg ollama
```

### Windows
Install these tools and add their executable directories to `PATH`:

- Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
- FFmpeg: https://ffmpeg.org/download.html
- Ollama: https://ollama.com/download

After installation, verify each command:

```bash
tesseract --version
pdftoppm -v
ffmpeg -version
ollama --version
```

## Configure Local AI
Start Ollama:

```bash
ollama serve
```

In another terminal, pull the default model:

```bash
ollama pull llama3.2:latest
```

The default model is configured as `llama3.2:latest`. You can use a different Ollama model by setting `OLLAMA_MODEL` in `.env`.

If Ollama is unavailable, the pipeline can still extract and preprocess text, but AI schema extraction may fall back to a text preview instead of structured JSON.

## Environment Configuration
Create a local `.env` file from the example if you need custom paths, model settings, OCR language, cache settings, or upload limits.

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Minimal working `.env` example:

```env
DATABASE_URL=sqlite:///data/local_ai.db
MAX_UPLOAD_SIZE_MB=500
OCR_LANGUAGE=eng
OCR_DPI=300
TESSERACT_CMD=tesseract
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
WHISPER_DEVICE=cpu
WHISPER_THREADS=4
FFMPEG_PATH=ffmpeg
CACHE_ENABLED=true
```

Most settings have defaults in `config.py`, so `.env` is optional unless your local paths or model settings differ.

Common settings:
- `DATABASE_URL` controls the SQLite database path.
- `MAX_UPLOAD_SIZE_MB` controls upload size limits.
- `OCR_LANGUAGE`, `OCR_DPI`, and `TESSERACT_CMD` control OCR behavior.
- `WHISPER_MODEL_DIR`, `WHISPER_DEVICE`, and `WHISPER_THREADS` control transcription.
- `LLM_PROVIDER`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL` control local AI inference.
- `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, and `CACHE_MAX_SIZE_MB` control caching.
- `FFMPEG_PATH` controls the FFmpeg executable used for video/audio processing.

Runtime directories are created automatically:
- `uploads/` for uploaded files
- `cache/` for OCR, transcription, and processing cache
- `data/` for SQLite database, search index, and exports
- `models/` for local model files
- `logs/` for application logs

## Validate Setup
Run the built-in environment check:

```bash
python app.py --check
```

The check must pass before the app launches successfully. If it fails, install the missing tool or fix the reported path/configuration issue.

## Run The Application
Start the Streamlit app:

```bash
python app.py
```

The launcher runs health checks first, then starts Streamlit using `ui/app.py`.

You can also run Streamlit directly:

```bash
streamlit run ui/app.py
```

Using `python app.py` is recommended because it validates the environment before launch.

## Basic Usage Workflow
1. Open the Streamlit UI after running `python app.py`.
2. Go to `Upload` and select one or more supported files.
3. Let the pipeline validate, store, extract, preprocess, and run AI schema extraction.
4. View extracted text and structured JSON in `Results`.
5. Use `Search` to find processed documents.
6. Use `History` to review previous uploads and processing states.
7. Use `Dashboard` to inspect processing and system status.
8. Export processed data using available export options.

## Processing Pipeline
The core pipeline is implemented in `services/pipeline_service.py`.

Flow:

```text
detect file type -> extract text -> preprocess text -> run local AI -> map schema -> store result
```

Extraction behavior:
- PDFs are routed through the PDF extractor, with OCR support for scanned content.
- Images are processed with Tesseract OCR.
- Audio files are transcribed with faster-whisper.
- Video files use FFmpeg plus transcription.
- Text, CSV, and JSON files are read directly.

## Exports
The export service supports:
- JSON
- CSV
- Excel, when `openpyxl` is installed
- Text reports
- ZIP bundles with structured output and optional original files

Exports are written under `data/exports/` by default.

## Development Setup
Use the same setup steps as normal usage, then install development tools:

```bash
pip install pytest openpyxl psutil
```

Recommended development loop:

```bash
python app.py --check
pytest
python app.py
```

Keep generated files such as uploads, caches, databases, logs, and model files out of commits unless they are intentionally required.

## Run Tests
Run all tests:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_ingestion.py
```

If `pytest` is not installed:

```bash
pip install pytest
```

Some tests or runtime features may require external tools such as Tesseract, Poppler, FFmpeg, or Ollama, depending on what is being exercised.

## Project Structure
- `app.py` - environment check and Streamlit launcher
- `config.py` - central settings loaded from environment variables and `.env`
- `ui/` - Streamlit pages, components, state, and styles
- `ingestion/` - file validation, storage, and deduplication
- `extraction/` - PDF, OCR, audio, video, and transcription extraction
- `preprocessing/` - text cleanup, metadata, features, OCR correction, and chunking
- `ai/` - local AI inference providers, prompts, retry, and streaming helpers
- `schema_mapping/` - schema extraction, schema definitions, validation, and repair
- `database/` - SQLAlchemy models, sessions, repository, and search support
- `services/` - pipeline, documents, cache, search, export, monitoring, and recovery
- `utils/` - logging, constants, exceptions, file utilities, and system checks
- `tests/` - automated tests
- `docs/` - process, growth, and expansion documentation
- `data/` - generated database, exports, and search index
- `uploads/` - uploaded files
- `cache/` - generated cache files
- `models/` - local model directories
- `logs/` - runtime logs

## Troubleshooting
### `python app.py --check` fails for Tesseract
Install Tesseract and verify `tesseract --version` works. If it is installed in a custom location, set `TESSERACT_CMD` in `.env`.

### `python app.py --check` fails for Poppler
Install Poppler and verify `pdftoppm -v` works. Poppler is required by `pdf2image` for PDF image conversion.

### `python app.py --check` fails for FFmpeg
Install FFmpeg and verify `ffmpeg -version` works. FFmpeg is required for audio/video workflows.

### `python app.py --check` fails for Ollama
Install Ollama and verify `ollama --version` works. Start Ollama with `ollama serve`, then pull the configured model with `ollama pull llama3.2:latest`.

### Streamlit does not start
Run `python app.py --check` first. Then confirm `streamlit` is installed with `pip show streamlit`.

### Excel export fails
Install `openpyxl`:

```bash
pip install openpyxl
```

### Tests fail because `pytest` is missing
Install `pytest`:

```bash
pip install pytest
```

### OCR output is poor
Use clearer scans, increase source image quality, verify the correct OCR language is installed, and adjust `OCR_LANGUAGE` or `OCR_DPI` in `.env`.

### AI schema output is empty or weak
Confirm Ollama is running, the configured model is pulled, and `OLLAMA_BASE_URL` points to the correct local Ollama server.

## Replication Checklist
A new user should be able to reproduce the project by completing this checklist:

1. Clone the repository.
2. Install Python 3.12 or newer.
3. Create and activate a virtual environment.
4. Install `requirements.txt`.
5. Install Tesseract, Poppler, FFmpeg, and Ollama.
6. Pull the configured Ollama model.
7. Create `.env` from `.env.example` if local overrides are needed.
8. Run `python app.py --check` until all checks pass.
9. Run `python app.py`.
10. Upload a supported file and verify extraction, processing, search, and export.

## Documentation
- `docs/FEEDBACK_PROCESS.md` explains how feedback is collected and converted into improvements.
- `docs/GROWTH_PLAN.md` outlines adoption and community growth goals.
- `docs/EXPANSION_PLAN.md` outlines institutional and geographical expansion goals.

## Privacy And Offline Use
Local AI Structify is designed for offline document processing. Uploaded files, extracted text, structured data, cache files, logs, database records, and exports remain on the local machine unless the user explicitly moves or shares them.

For fully offline use, install all external tools and local models before disconnecting from the internet.

## Status
Current UI version shown in the app is `v0.1.0`. The project is under active development and should be tested with representative documents before production use.
