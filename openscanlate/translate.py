from __future__ import annotations

from typing import List, Optional
import os
import ollama


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        # Allow override via env var OLLAMA_HOST (e.g., http://localhost:11434)
        host = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=str(host))

    def ping(self, timeout: float = 2.0) -> bool:
        try:
            # Listing models is lightweight and confirms reachability
            _ = self.client.list()
            return True
        except Exception:
            return False

    def generate(self, model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        if system:
            # Use chat API to supply a system message explicitly
            resp = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": temperature},
            )
            msg = resp.get("message", {})
            return str(msg.get("content", "")).strip()
        else:
            resp = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature},
            )
            return str(resp.get("response", "")).strip()


def translate_chunks(client: OllamaClient, model: str, chunks: List[str], system: Optional[str] = None) -> List[str]:
    outputs: List[str] = []
    for jp in chunks:
        # Simple prompt; can be expanded with context
        if jp:
            prompt = (
                "Translate the following dialogue into natural English."
                " Keep honorifics when appropriate.\n\n"
                f"To Be Translated:\n{jp}\n\nEnglish:\n"
                "Dont mention anything else. Just Return the translation without any additional commentary."
            )
            out = client.generate(model=model, prompt=prompt, system=system)
            outputs.append(out)
    return outputs
