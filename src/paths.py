from pathlib import Path


def source_name_from_pdf(
    pdf_path: Path,
) -> str:
    return Path(pdf_path).stem


def image_directory_for_pdf(
    pdf_path: Path,
) -> Path:
    return (
        Path("data/images")
        / source_name_from_pdf(pdf_path)
    )


def paddle_directory_for_pdf(
    pdf_path: Path,
) -> Path:
    return (
        Path("data/ocr/paddle")
        / source_name_from_pdf(pdf_path)
    )
