"""Document loading, section parsing, chunking, and benchmark loading."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .schemas import BenchmarkCase, DocumentChunk

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


class CorpusLoader:
    def __init__(self, max_chars: int = 500, overlap: int = 80) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if overlap < 0 or overlap >= max_chars:
            raise ValueError("overlap must be >= 0 and less than max_chars")
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self, root: Path) -> list[DocumentChunk]:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")

        chunks: list[DocumentChunk] = []
        files = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
            key=lambda path: path.as_posix().lower(),
        )
        for path in files:
            chunks.extend(self._load_file(path))
        return chunks

    def _load_file(self, path: Path) -> list[DocumentChunk]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")

        sections = self._extract_sections(text, path.stem)
        chunks: list[DocumentChunk] = []
        sequence = 1
        for section, section_text in sections:
            for part in self._split_text(section_text):
                chunks.append(
                    DocumentChunk(
                        doc_id=path.stem,
                        title=path.stem,
                        source=path.as_posix(),
                        section=section,
                        chunk_id=f"{path.stem}::chunk-{sequence:04d}",
                        text=part,
                    )
                )
                sequence += 1
        return chunks

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("PDF support requires PyMuPDF (fitz)") from exc

        document = fitz.open(path)
        try:
            return "\n\n".join(page.get_text("text") for page in document)
        finally:
            document.close()

    @staticmethod
    def _extract_sections(text: str, default_title: str) -> list[tuple[str, str]]:
        current_section = default_title
        current_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_section, content))

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            heading = re.match(r"^#{1,6}\s+(.+)$", line)
            if heading:
                flush()
                current_lines.clear()
                current_section = heading.group(1).strip()
                continue
            current_lines.append(line)
        flush()
        return sections or [(default_title, text.strip())]

    def _split_text(self, text: str) -> list[str]:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) <= self.max_chars:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end))
                if boundary > start + self.max_chars // 2:
                    end = boundary + 1
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
        return parts


def load_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                case = BenchmarkCase(**payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid benchmark case at line {line_number}: {exc}") from exc
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate benchmark case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases
