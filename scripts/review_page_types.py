import argparse

import matplotlib.pyplot as plt
from PIL import Image

from src.database import get_connection
from src.paths import from_project_relative


def get_unreviewed_sketches(
    connection,
    *,
    source_file_id: int,
):
    """
    Return pages algorithmically tagged as sketches
    but not yet reviewed by a human.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                page_id,
                source_page_number,
                corpus_page_number,
                image_path,
                report_id,
                report_page_number
            FROM page
            WHERE
                source_file_id = %s
                AND page_type = 'sketch'
                AND page_type_reviewed = FALSE
            ORDER BY source_page_number;
            """,
            (source_file_id,),
        )

        return cursor.fetchall()


def show_page(
    *,
    image_path,
    source_page_number: int,
):
    """
    Display the complete rendered page for human review.

    Returns:
        (figure, image)
    """

    resolved_path = from_project_relative(
        image_path
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Page image not found: "
            f"{resolved_path}"
        )

    image = Image.open(
        resolved_path
    )

    # Scale figure approximately to page aspect ratio.
    aspect_ratio = (
        image.width
        / image.height
    )

    figure_height = 9

    figure_width = (
        figure_height
        * aspect_ratio
    )

    fig, ax = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    ax.imshow(
        image
    )

    ax.set_title(
        f"Source page "
        f"{source_page_number:04d}"
    )

    ax.axis(
        "off"
    )

    plt.tight_layout()

    # Non-blocking so terminal input remains usable.
    plt.show(
        block=False
    )

    return (
        fig,
        image,
    )


def close_page(
    display,
):
    """
    Close Matplotlib/Pillow resources.
    """

    if display is None:
        return

    fig, image = display

    plt.close(
        fig
    )

    image.close()


def confirm_sketch(
    connection,
    *,
    page_id: int,
):
    """
    Human confirms that this page is a sketch.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE page
            SET
                page_type = 'sketch',
                page_type_reviewed = TRUE
            WHERE page_id = %s;
            """,
            (page_id,),
        )


def reject_sketch(
    connection,
    *,
    page_id: int,
):
    """
    Human determines that this is not a sketch.

    It returns to 'unknown' rather than 'report'
    because report segmentation may not have happened yet.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE page
            SET
                page_type = 'unknown',
                page_type_reviewed = TRUE
            WHERE page_id = %s;
            """,
            (page_id,),
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Review pages automatically tagged "
            "as probable sketches."
        )
    )

    parser.add_argument(
        "--source-file",
        type=int,
        required=True,
        help="source_file_id to review",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Verify source + retrieve candidates
    # ---------------------------------------------------------

    with get_connection() as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT filename
                FROM source_file
                WHERE source_file_id = %s;
                """,
                (args.source_file,),
            )

            source = cursor.fetchone()

        if source is None:
            raise RuntimeError(
                f"Source file ID "
                f"{args.source_file} "
                f"does not exist."
            )

        filename = source[0]

        pages = get_unreviewed_sketches(
            connection,
            source_file_id=args.source_file,
        )

    # ---------------------------------------------------------
    # Nothing to review
    # ---------------------------------------------------------

    if not pages:
        print()
        print(
            "No unreviewed probable sketches found."
        )
        return

    total = len(
        pages
    )

    print()
    print("=== SKETCH PAGE REVIEW ===")
    print()

    print(
        f"Source:     {filename}"
    )

    print(
        f"Candidates: {total}"
    )

    print()

    confirmed = 0
    rejected = 0
    skipped = 0

    # =========================================================
    # Review loop
    # =========================================================

    for index, (
        page_id,
        source_page_number,
        corpus_page_number,
        image_path,
        report_id,
        report_page_number,
    ) in enumerate(
        pages,
        start=1,
    ):

        print()
        print("=" * 70)
        print()

        print(
            f"Candidate "
            f"{index} / {total}"
        )

        print(
            f"Page ID:        "
            f"{page_id}"
        )

        print(
            f"Source page:    "
            f"{source_page_number:04d}"
        )

        print(
            f"Corpus page:    "
            f"{corpus_page_number}"
        )

        if report_id is not None:
            print(
                f"Report ID:      "
                f"{report_id}"
            )

            print(
                f"Report page:    "
                f"{report_page_number}"
            )

        print()
        print(
            "Algorithm classification: "
            "probable sketch"
        )
        print()

        # -----------------------------------------------------
        # Open original page
        # -----------------------------------------------------

        display = show_page(
            image_path=image_path,
            source_page_number=(
                source_page_number
            ),
        )

        print(
            "[y] confirm sketch"
        )

        print(
            "[n] not a sketch"
        )

        print(
            "[s] skip for now"
        )

        print(
            "[q] quit"
        )

        print()

        choice = input(
            "> "
        ).strip().casefold()

        # -----------------------------------------------------
        # Handle decision
        #
        # Keep image open until AFTER the user has entered
        # the decision.
        # -----------------------------------------------------

        should_quit = False

        match choice:

            case "y":

                with get_connection() as connection:

                    confirm_sketch(
                        connection,
                        page_id=page_id,
                    )

                    connection.commit()

                confirmed += 1

                print(
                    "Confirmed as sketch."
                )

            case "n":

                with get_connection() as connection:

                    reject_sketch(
                        connection,
                        page_id=page_id,
                    )

                    connection.commit()

                rejected += 1

                print(
                    "Rejected sketch classification."
                )

            case "s":

                skipped += 1

                print(
                    "Left unreviewed."
                )

            case "q":

                should_quit = True

            case _:

                skipped += 1

                print(
                    "Unknown command. "
                    "Page left unreviewed."
                )

        # One common image-cleanup point.
        close_page(
            display
        )

        if should_quit:

            print()
            print(
                "Exiting sketch review."
            )

            break

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("--- Review Summary ---")
    print()

    print(
        f"Confirmed sketches: {confirmed}"
    )

    print(
        f"Rejected:           {rejected}"
    )

    print(
        f"Skipped/unreviewed: {skipped}"
    )

if __name__ == "__main__":
    main()
