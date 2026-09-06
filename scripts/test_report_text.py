import argparse

from src.corpus_text import get_report_text


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "report_id",
        type=int,
    )

    args = parser.parse_args()

    text = get_report_text(
        args.report_id
    )

    print(text)


if __name__ == "__main__":
    main()
