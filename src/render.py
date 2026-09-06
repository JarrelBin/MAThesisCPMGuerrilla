from pathlib import Path

import fitz


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
    overwrite: bool = False,
) -> list[Path]:
    """
    Render every PDF page to PNG.

    Output:
        page_0001.png
        page_0002.png
        ...

    Returns the ordered list of page image paths.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = fitz.open(pdf_path)

    rendered_pages = []

    try:

        for page_index in range(
            document.page_count
        ):

            page_number = page_index + 1

            output_path = (
                output_dir
                / f"page_{page_number:04d}.png"
            )

            rendered_pages.append(
                output_path
            )

            if (
                output_path.exists()
                and not overwrite
            ):
                print(
                    f"[SKIP] page_{page_number:04d}: "
                    "image already exists"
                )

                continue

            page = document.load_page(
                page_index
            )

            pixmap = page.get_pixmap(
                dpi=dpi,
                alpha=False,
            )

            pixmap.save(
                output_path
            )

            print(
                f"[REND] page_{page_number:04d}"
            )

    finally:
        document.close()

    return rendered_pages
