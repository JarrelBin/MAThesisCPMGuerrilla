def normalize_analysis_token(
    token: str,
) -> str:
    """
    Normalize a token for analytical use.

    Does NOT modify PostgreSQL.
    """

    return token.casefold()


def preprocess_tokens(
    tokens: list[str],
    *,
    lowercase: bool = True,
) -> list[str]:

    result = []

    for token in tokens:

        if lowercase:
            token = (
                normalize_analysis_token(
                    token
                )
            )

        result.append(token)

    return result
