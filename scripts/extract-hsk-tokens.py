"""
Extract DeepSeek V3 token IDs for HSK character ranges.

Builds per-level logit-bias maps so the simplifier can softly bias the LLM
toward only HSK-level-appropriate vocabulary.

Outputs:
  data/hsk-token-ids-1.json       (HSK 1 only)
  data/hsk-token-ids-1to2.json    (HSK 1-2)
  data/hsk-token-ids-1to3.json    (HSK 1-3 — same as legacy hsk-token-ids.json)
  data/hsk-token-ids-1to4.json    (HSK 1-4)

Each file shape:
  {
    "char_to_tokens": { "我": [1234], "你": [5678], ... },
    "single_token_chars": [...],
    "multi_token_chars": [...],
    "single_char_token_ids": [...],
    "multi_char_hsk_pure_ids": [...],
    "all_unique_token_ids": [...],
    "total_unique_tokens": N,
    "tokenizer_repo": "deepseek-ai/DeepSeek-V3",
    "hsk_levels": [1, 2, ...]
  }
"""
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATA = Path(__file__).parent.parent / "data"

# Tokenizer repo + output filename suffix are configurable so the same script
# can build token-ID maps for any model. Defaults reproduce the DeepSeek files.
#   TOKENIZER_REPO=openai/gpt-oss-120b   TOKEN_SUFFIX=.gpt-oss
#   TOKENIZER_REPO=MiniMaxAI/MiniMax-M2.5 TOKEN_SUFFIX=.minimax
REPO = os.environ.get("TOKENIZER_REPO", "deepseek-ai/DeepSeek-V3")
SUFFIX = os.environ.get("TOKEN_SUFFIX", "")


def load_hsk_chars(levels):
    """Collect all unique Chinese chars from HSK word lists for the given levels."""
    chars = set()
    for level in levels:
        words = json.loads((DATA / f"hsk{level}.json").read_text(encoding="utf-8"))
        for w in words:
            for ch in w:
                if "一" <= ch <= "鿿":  # CJK Unified Ideographs
                    chars.add(ch)
    return sorted(chars)


def build_for_levels(levels, tokenizer, output_path):
    """Build a token-IDs file covering the given HSK levels."""
    hsk_chars = load_hsk_chars(levels)
    print(f"\n=== HSK {min(levels)}-{max(levels)} ({len(hsk_chars)} unique chars) ===")
    hsk_char_set = set(hsk_chars)

    char_to_tokens = {}
    single_token = []
    multi_token = []

    for ch in hsk_chars:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        ids = enc.ids
        char_to_tokens[ch] = ids
        if len(ids) == 1:
            single_token.append(ch)
        else:
            multi_token.append(ch)

    all_ids = set()
    for ids in char_to_tokens.values():
        all_ids.update(ids)

    print(f"  Single-token chars: {len(single_token)}")
    print(f"  Multi-token chars:  {len(multi_token)}")
    print(f"  Single-char-based token IDs: {len(all_ids)}")

    # Scan full vocab for multi-char HSK-pure tokens.
    # A token is HSK-pure if every Chinese char in its decoded string is in our HSK char set.
    vocab = tokenizer.get_vocab()
    multi_char_hsk_pure_ids = set()

    for token_str, token_id in vocab.items():
        decoded = tokenizer.decode([token_id])
        chinese_chars = [c for c in decoded if "一" <= c <= "鿿"]
        if len(chinese_chars) < 2:
            continue
        if all(c in hsk_char_set for c in chinese_chars):
            multi_char_hsk_pure_ids.add(token_id)

    combined_ids = all_ids | multi_char_hsk_pure_ids

    print(f"  Multi-char HSK-pure tokens: {len(multi_char_hsk_pure_ids)}")
    print(f"  TOTAL bias entries:         {len(combined_ids)}")

    output_path.write_text(json.dumps({
        "char_to_tokens": char_to_tokens,
        "single_token_chars": single_token,
        "multi_token_chars": multi_token,
        "single_char_token_ids": sorted(all_ids),
        "multi_char_hsk_pure_ids": sorted(multi_char_hsk_pure_ids),
        "all_unique_token_ids": sorted(combined_ids),
        "total_unique_tokens": len(combined_ids),
        "tokenizer_repo": REPO,
        "hsk_levels": list(levels),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wrote {output_path}")


def main():
    # TOKENIZER_FILE loads a local tokenizer.json (e.g. data/dsv4-tokenizer.json)
    # instead of downloading from HF hub.
    local = os.environ.get("TOKENIZER_FILE")
    if local:
        print(f"Loading local tokenizer {local}...")
        path = local
    else:
        print(f"Downloading tokenizer.json from {REPO}...")
        path = hf_hub_download(repo_id=REPO, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    print(f"Loaded tokenizer (vocab size {tokenizer.get_vocab_size()})")

    # LEVELS env selects which cumulative maps to (re)build, e.g. LEVELS=5,6.
    # Default reproduces the original 1-4 set.
    max_levels = [int(x) for x in os.environ.get("LEVELS", "1,2,3,4").split(",")]
    for n in max_levels:
        levels = list(range(1, n + 1))
        name = "1" if n == 1 else f"1to{n}"
        build_for_levels(levels, tokenizer, DATA / f"hsk-token-ids-{name}{SUFFIX}.json")

    print("\nDone. Note: data/hsk-token-ids.json (legacy HSK 1-3) is preserved as fallback.")


if __name__ == "__main__":
    main()
