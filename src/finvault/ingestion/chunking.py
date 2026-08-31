"""Paragraph-aware chunking with trailing-context overlap."""

from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Splits on paragraph boundaries, then packs paragraphs into chunks up to
    `chunk_size` characters. `overlap` characters of trailing context from the
    end of one chunk are carried into the start of the next, so a fact split
    across a chunk boundary isn't lost entirely from either side.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    sub_paragraphs: list[str] = []
    for para in paragraphs:
        if len(para) > chunk_size:
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            for line in lines:
                if len(line) > chunk_size:
                    for i in range(0, len(line), chunk_size):
                        sub_paragraphs.append(line[i : i + chunk_size])
                else:
                    sub_paragraphs.append(line)
        else:
            sub_paragraphs.append(para)

    chunks: list[str] = []
    current = ""
    for para in sub_paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{para}" if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks
