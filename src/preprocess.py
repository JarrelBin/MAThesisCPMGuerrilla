from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageOps

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
    

def binarize_image(
    input_path: Path,
    output_path: Path,
    threshold: int = 180
) -> Path:
    """
    Convert an image to a bi-level (1-bit) image.

    The original image is not modified.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(input_path) as image:

        # Convert to grayscale
        grayscale = ImageOps.grayscale(image)

        # Convert pixels to pure black or white
        binary = grayscale.point(
            lambda pixel: 255 if pixel > threshold else 0,
            mode="1"
        )

        binary.save(output_path)

    return output_path
