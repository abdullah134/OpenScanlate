from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import typer
from rich import print
from rich.progress import Progress

from .ocr import detect_and_recognize
from .inpaint import inpaint_regions
from .translate import OllamaClient, translate_chunks
from .typeset import paste_texts

app = typer.Typer(help="OpenScanlate Phase 1 CLI")


@app.command()
def run(
    image: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="Input image path"),
    out: Path = typer.Option(Path("out"), help="Output directory"),
    model: str = typer.Option("llama3", help="Ollama model name"),
    prompt: Optional[str] = typer.Option(None, help="Optional system/context prompt"),
    font: Optional[Path] = typer.Option(None, exists=True, help="Path to a .ttf/.otf font for typesetting"),
    font_size: int = typer.Option(28, help="Font size for typesetting"),
    use_manga_ocr: bool = typer.Option(False, help="Use Manga-OCR for recognition if installed"),
):
    out.mkdir(parents=True, exist_ok=True)

    # Validate font early so we fail fast
    if font is None:
        print("[yellow]--font is required for typesetting. Provide a TTF/OTF font path.[/]")
        raise typer.Exit(code=2)

    # Ensure Ollama is reachable before heavy work
    client = OllamaClient()
    if not client.ping():
        print("[red]Can't reach Ollama at http://localhost:11434. Start Ollama and pull a model (e.g., llama3).[/]")
        print("Tip: set OLLAMA_HOST if Ollama runs on a different host/port.")
        raise typer.Exit(code=3)

    print(f"[bold]Loading image:[/] {image}")
    pil = Image.open(image).convert("RGB")
    np_img = np.array(pil)

    with Progress() as prog:
        t_ocr = prog.add_task("OCR", total=1)
        ocr_boxes = detect_and_recognize(np_img, use_manga_ocr=use_manga_ocr)
        prog.advance(t_ocr)

    # Save OCR results
    ocr_data = [
        {"box": list(b.box), "text": b.text, "score": b.score} for b in ocr_boxes
    ]
    (out / "ocr.json").write_text(json.dumps(ocr_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if not ocr_boxes:
        print("[yellow]No text detected. Exiting.[/]")
        raise typer.Exit(code=0)

    with Progress() as prog:
        t_clean = prog.add_task("Inpainting", total=1)
        cleaned_np = inpaint_regions(np_img, [b.box for b in ocr_boxes])
        prog.advance(t_clean)

    cleaned_pil = Image.fromarray(cleaned_np)
    cleaned_path = out / "cleaned_image.png"
    cleaned_pil.save(cleaned_path)

    with Progress() as prog:
        t_tx = prog.add_task("Translating", total=len(ocr_boxes))
        translations = translate_chunks(client, model, [b.text for b in ocr_boxes], system=prompt)
        for _ in translations:
            prog.advance(t_tx)

    # Save translations
    (out / "translations.json").write_text(
        json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Typeset
    final_img = paste_texts(cleaned_pil, [b.box for b in ocr_boxes], translations, str(font), font_size)
    final_path = out / "final_image.png"
    final_img.save(final_path)

    print("[green]Done.[/]")
    print(f"Cleaned: {cleaned_path}")
    print(f"Final:   {final_path}")


if __name__ == "__main__":
    app()
