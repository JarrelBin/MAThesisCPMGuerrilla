from pathlib import Path


# src/paths.py
#
# __file__:
#   <project>/src/paths.py
#
# parent:
#   <project>/src
#
# parent.parent:
#   <project>

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


RAW_DIR = (
    DATA_DIR
    / "raw"
)


IMAGE_DIR = (
    DATA_DIR
    / "images"
)


OCR_DIR = (
    DATA_DIR
    / "ocr"
)


PADDLE_OCR_DIR = (
    OCR_DIR
    / "paddle"
)


GROUND_TRUTH_DIR = (
    DATA_DIR
    / "ground_truth"
)


def source_name(
    path: Path | str,
) -> str:
    """
    Return a source name without its extension.

    Works with Windows and POSIX paths.
    """

    return Path(path).stem


def image_directory(
    source: Path | str,
) -> Path:
    """
    Page-image directory for a source PDF.
    """

    return (
        IMAGE_DIR
        / source_name(source)
    )


def paddle_directory(
    source: Path | str,
) -> Path:
    """
    PaddleOCR output directory for a source PDF.
    """

    return (
        PADDLE_OCR_DIR
        / source_name(source)
    )
    
def to_project_relative(
    path: Path | str,
) -> str:
    """
    Convert a project file path into a portable,
    project-relative POSIX-style path for database
    storage.

    Example:

        data/images/source/page_0001.png
    """

    path = Path(path).resolve()

    relative = path.relative_to(
        PROJECT_ROOT
    )

    return relative.as_posix()


def from_project_relative(
    path: Path | str,
) -> Path:
    """
    Resolve a database project-relative path on the
    current operating system.
    """

    return (
        PROJECT_ROOT
        / Path(path)
    ).resolve()
