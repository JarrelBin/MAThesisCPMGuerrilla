from pathlib import Path
from pdf2image import convert_from_path


def pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300
) -> list[Path]:

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF at {dpi} DPI...")

    pages = convert_from_path(
        pdf_path,
        dpi=dpi
    )

    image_paths = []

    for page_number, page in enumerate(pages, start=1):

        output_path = (
            output_dir / f"page_{page_number:04d}.png"
        )

        page.save(output_path, "PNG")

        image_paths.append(output_path)

        print(
            f"Saved page {page_number}: "
            f"{output_path}"
        )

    return image_paths
