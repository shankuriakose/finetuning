import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")
TOKENIZER_FILES = ("tokenizer_config.json",)
TOKENIZER_FILE_GLOBS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def _remove_tokenizer_files(local_dir: Path) -> None:
    for pattern in TOKENIZER_FILE_GLOBS:
        for f in local_dir.glob(pattern):
            f.unlink()


def _exists(local_dir: Path, filenames=(), globs=()) -> bool:
    if any((local_dir / f).exists() for f in filenames):
        return True
    return any(next(local_dir.glob(g), None) is not None for g in globs)


def download_model(model_name: str, force: bool = False, force_tokenizer: bool = False,
                    base_dir: Path | str = ".") -> Path:
    folder_name = model_name.replace("/", "__")
    local_dir = Path(base_dir) / folder_name
    local_dir.mkdir(parents=True, exist_ok=True)

    if force or not _exists(local_dir, globs=MODEL_WEIGHT_GLOBS):
        config_8bit = BitsAndBytesConfig(load_in_8bit=True)

        model_8bit = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config_8bit,
            device_map="auto",
            trust_remote_code=True
        )
        model_8bit.save_pretrained(local_dir)
    else:
        print(f"Model weights already exist in {local_dir}, skipping download.")

    if force_tokenizer or force or not _exists(local_dir, filenames=TOKENIZER_FILES):
        if force_tokenizer or force:
            _remove_tokenizer_files(local_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        tokenizer.save_pretrained(local_dir)
    else:
        print(f"Tokenizer already exists in {local_dir}, skipping download.")

    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download a model from the Hugging Face Hub")
    parser.add_argument("model_name", nargs="?", default="meta-llama/Llama-3.2-1B-Instruct",
                         help="e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--force", action="store_true",
                         help="Re-download model weights and tokenizer even if they already exist")
    parser.add_argument("--force-tokenizer", action="store_true",
                         help="Re-download the tokenizer even if it already exists")
    parser.add_argument("--base-dir", default=".",
                         help="Directory under which the model folder is created (e.g. a mounted Drive path)")
    args = parser.parse_args()

    local_dir = download_model(args.model_name, force=args.force, force_tokenizer=args.force_tokenizer,
                                base_dir=args.base_dir)
    print(f"Downloaded '{args.model_name}' to {local_dir.resolve()}")


if __name__ == "__main__":
    main()
