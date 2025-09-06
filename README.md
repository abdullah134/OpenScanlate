# OpenScanlate (Phase 1 CLI)

A local-first pipeline to translate manga pages using OCR, inpainting, Ollama for translation, and simple typesetting.

This is the Phase 1 command-line tool proving the core pipeline works on a single image. A GUI will come in later phases.

## Features (Phase 1)

- OCR with bounding boxes (EasyOCR for detection by default; optional Manga-OCR for recognition if installed)
- Inpainting of original text regions using OpenCV
- Translation via a local Ollama model (default: llama3)
- Typesetting into original bubbles using Pillow, with simple word wrapping

## Prerequisites

- Python 3.9–3.11 (64-bit recommended)
- Windows supported (this repo was created on Windows). Linux/macOS should also work with minor tweaks.
- Ollama installed and running locally (https://ollama.com). Pull at least one model, e.g. `llama3`.
- GPU not required (EasyOCR will run on CPU by default), but optional.

## Install

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

Notes:
- EasyOCR pulls PyTorch wheels which are large; installation can take a while.
- If you want Manga-OCR recognition (optional), also `pip install manga-ocr` (listed in requirements as an extra). It may require additional model downloads on first run.

3) Ensure Ollama is running and you have a model ready

```bash
ollama pull llama3
```

## Usage

Translate a single image and produce `final_image.png` in the output folder.

```bash
python -m openscanlate.cli \
  --image path/to/page.png \
  --out out \
  --model llama3 \
  --font path/to/AnimeAce.ttf \
  --font-size 28
```

Flags:
- `--use-manga-ocr`: If provided and `manga-ocr` is installed, uses EasyOCR for detection and Manga-OCR for recognition per detected box.
- `--prompt`: Optional translation style/context. Example: "You are a manga translator. Keep honorifics."

Outputs in the `--out` directory:
- `cleaned_image.png` – image after inpainting speech bubbles
- `final_image.png` – typeset result
- `ocr.json` – JSON with detected boxes and text
- `translations.json` – JSON with translated texts

## Known limits in Phase 1

- Box detection uses EasyOCR; quality varies. Phase 3 may replace with dedicated detectors or refined pipelines.
- Text layout is basic; future phases will add better fitting, alignment, and interactive tweaking.
- No batch/CBZ support yet; that will come in Phase 3.

## Roadmap

- Phase 2: Basic GUI (PySide6/PyQt6 or CustomTkinter)
- Phase 3: Interactivity, project save/load, better inpainting, batch processing

## License

MIT
# OpenScanlate
It is open source tool for translator and people read manga,manhwa and mahuwa. translate it using google translate or AI. The goal is to make it easier for people to translate raw file. Make it accessable to everyone
