import os


def resolve_tokenizer_subdir(model_path: str, tokenizer_path: str | None = None) -> str:
    """Infer tokenizer subdirectory for multimodal models.

    Returns a subdirectory name to pass to get_tokenizer, or "" if unknown.
    """
    if not model_path:
        return ""

    candidate_path = tokenizer_path or model_path
    if not candidate_path:
        return ""

    normalized = candidate_path.rstrip("/")
    basename = os.path.basename(normalized)
    if basename.startswith("tokenizer"):
        return ""
    if basename.startswith("text_encoder"):
        # Model path already points to a component; do not append.
        return ""

    if os.path.isdir(normalized):
        tokenizer_dir = os.path.join(normalized, "tokenizer")
        if os.path.isdir(tokenizer_dir):
            return "tokenizer"
        tokenizer_dir = os.path.join(normalized, "tokenizer_2")
        if os.path.isdir(tokenizer_dir):
            return "tokenizer_2"
        return ""

    if "diffusers" in model_path.lower():
        return "tokenizer"

    return ""
