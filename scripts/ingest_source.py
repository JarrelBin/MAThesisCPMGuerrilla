import argparse
from pathlib import Path

from PIL import Image

from src.database import get_connection
from src.ingest import inspect_source_file
from src.paths import image_directory_for_pdf
from src.render import render_pdf
from src.repository import (
    get_or_create_corpus,
    get_source_file_by_sha256,
    insert_page,
    insert_source_file,
    recalculate_corpus_page_numbers,
)


CORPUS_NAME = "Defector Reports"
DPI = 300


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render and register an archival "
            "PDF source file."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to source PDF",
    )

    parser.add_argument(
        "--sequence",
        type=int,
        required=True,
        help=(
            "Physical ordering of this PDF "
            "within the corpus"
        ),
    )

    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help=(
            "Regenerate existing page images"
        ),
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {args.pdf}"
        )

    # =========================================================
    # Inspect source file
    # =========================================================

    metadata = inspect_source_file(
        args.pdf
    )

    # =========================================================
    # Render PDF
    # =========================================================

    image_dir = image_directory_for_pdf(
        args.pdf
    )

    print()
    print("=== PDF RENDERING ===")
    print()

    print(
        f"Source: {args.pdf}"
    )

    print(
        f"Output: {image_dir}"
    )

    print(
        f"DPI:    {DPI}"
    )

    print()

    page_images = render_pdf(
        args.pdf,
        image_dir,
        dpi=DPI,
        overwrite=args.overwrite_images,
    )

    if not page_images:
        raise RuntimeError(
            "PDF contains no pages."
        )

    page_count = len(
        page_images
    )

    # =========================================================
    # Database
    # =========================================================

    with get_connection() as connection:

        corpus_id = get_or_create_corpus(
            connection,
            name=CORPUS_NAME,
            description=(
                "Historical defector reports "
                "contained across archival PDF "
                "source files."
            ),
        )

        # -----------------------------------------------------
        # Idempotency
        # -----------------------------------------------------

        existing = get_source_file_by_sha256(
            connection,
            metadata["sha256"],
        )

        if existing is not None:

            connection.commit()

            print()
            print(
                "Source file already registered."
            )

            print(
                f"Source file ID: {existing[0]}"
            )

            return

        # -----------------------------------------------------
        # Source file
        # -----------------------------------------------------

        source_file_id = insert_source_file(
            connection,
            corpus_id=corpus_id,
            filename=metadata["filename"],
            source_path=metadata["filepath"],
            sha256=metadata["sha256"],
            page_count=page_count,
            file_sequence=args.sequence,
        )

        # -----------------------------------------------------
        # Temporary corpus page numbers
        #
        # These are recalculated after all pages are inserted.
        # -----------------------------------------------------

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT COALESCE(
                    MAX(corpus_page_number),
                    0
                )
                FROM page;
                """
            )

            next_corpus_page = (
                cursor.fetchone()[0] + 1
            )

        # -----------------------------------------------------
        # Pages
        # -----------------------------------------------------

        for (
            source_page_number,
            image_path,
        ) in enumerate(
            page_images,
            start=1,
        ):

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Rendered image missing: "
                    f"{image_path}"
                )

            with Image.open(
                image_path
            ) as image:

                width, height = (
                    image.size
                )

            page_id = insert_page(
                connection,
                source_file_id=source_file_id,
                source_page_number=(
                    source_page_number
                ),
                corpus_page_number=(
                    next_corpus_page
                ),
                image_path=str(image_path),
                width=width,
                height=height,
                dpi=DPI,
            )

            print(
                f"[ DB ] "
                f"page_{source_page_number:04d} "
                f"-> page_id={page_id}"
            )

            next_corpus_page += 1

        # -----------------------------------------------------
        # Restore correct corpus-wide ordering according to:
        #
        # source_file.file_sequence
        # +
        # page.source_page_number
        # -----------------------------------------------------

        recalculate_corpus_page_numbers(
            connection
        )

        connection.commit()

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("--- Summary ---")
    print()

    print(
        f"Source file ID: {source_file_id}"
    )

    print(
        f"Sequence:       {args.sequence}"
    )

    print(
        f"Pages:          {page_count}"
    )

    print(
        f"Image directory:"
        f" {image_dir}"
    )

    print()
    print(
        "Source ingestion complete."
    )


if __name__ == "__main__":
    main()
