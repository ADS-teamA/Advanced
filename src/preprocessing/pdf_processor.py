"""
PDF processing interfaces and implementations.

Provides pluggable PDF text extraction with PyPDF2 (MVP)
and placeholder for future OCR support (DiT + EasyOCR).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import PyPDF2

from src.utils.errors import PDFProcessingError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFProcessor(ABC):
    """Abstract base class for PDF processors."""

    @abstractmethod
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content

        Raises:
            PDFProcessingError: If extraction fails
        """
        pass


class PyPDFProcessor(PDFProcessor):
    """
    PDF processor using PyPDF2 for text extraction.

    This is the MVP implementation that handles most text-based PDFs.
    It does not support scanned/image-based PDFs (use OCRProcessor for those).
    """

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content

        Raises:
            PDFProcessingError: If PDF is corrupted or unreadable
        """
        try:
            pdf_file_path = Path(pdf_path)

            if not pdf_file_path.exists():
                raise PDFProcessingError(f"PDF file not found: {pdf_path}")

            if not pdf_file_path.suffix.lower() == '.pdf':
                raise PDFProcessingError(f"File is not a PDF: {pdf_path}")

            logger.info(f"Extracting text from PDF: {pdf_path}")

            with open(pdf_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {pdf_path}")
                    try:
                        pdf_reader.decrypt('')
                    except Exception as e:
                        raise PDFProcessingError(
                            f"Failed to decrypt PDF {pdf_path}: {e}"
                        )

                # Extract text from all pages
                text_parts = []
                total_pages = len(pdf_reader.pages)

                logger.info(f"Processing {total_pages} pages...")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                        if page_num % 10 == 0:
                            logger.debug(f"Processed {page_num}/{total_pages} pages")

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                        continue

                extracted_text = "\n\n".join(text_parts)

                if not extracted_text.strip():
                    raise PDFProcessingError(
                        f"No text extracted from PDF {pdf_path}. "
                        "The PDF may be image-based (scanned). Consider using OCR."
                    )

                logger.info(
                    f"Successfully extracted {len(extracted_text)} characters "
                    f"from {total_pages} pages"
                )

                return extracted_text

        except PyPDF2.errors.PdfReadError as e:
            raise PDFProcessingError(
                f"Failed to read PDF {pdf_path}: {e}. File may be corrupted."
            ) from e

        except Exception as e:
            raise PDFProcessingError(
                f"Unexpected error processing PDF {pdf_path}: {e}"
            ) from e


class OCRProcessor(PDFProcessor):
    """
    PDF processor using OCR for scanned documents.

    FUTURE IMPLEMENTATION: Will use DiT (Document Image Transformer)
    + EasyOCR as specified in PIPELINE.md (Narendra et al., 2024).

    This is a placeholder for future development.
    """

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text using OCR (not yet implemented).

        Args:
            pdf_path: Path to PDF file

        Raises:
            NotImplementedError: OCR support coming soon
        """
        raise NotImplementedError(
            "OCR support is not yet implemented. "
            "Future version will use DiT + EasyOCR for scanned PDFs. "
            "For now, use PyPDFProcessor for text-based PDFs."
        )


def get_pdf_processor(use_ocr: bool = False) -> PDFProcessor:
    """
    Factory function to get the appropriate PDF processor.

    Args:
        use_ocr: If True, return OCRProcessor (future). If False, return PyPDFProcessor.

    Returns:
        PDFProcessor instance
    """
    if use_ocr:
        return OCRProcessor()
    return PyPDFProcessor()
