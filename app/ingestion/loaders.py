from pathlib import Path

from pypdf import PdfReader

from app.ingestion.metadata import ParsedPage


def load_pdf(path: Path) -> list[ParsedPage]:
    """Extract PDF text independently per one-indexed page."""
    try:
        reader = PdfReader(path)
    except Exception as error:
        raise ValueError(f"Unable to open PDF: {path}") from error

    pages: list[ParsedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise ValueError(f"Unable to extract page {page_number} from {path}") from error
        pages.append(ParsedPage(page_number=page_number, text=text))
    return pages
