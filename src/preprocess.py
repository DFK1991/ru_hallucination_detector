import re

# compile outside once
NUM_REGEX = re.compile(r'\b\d+(?:[.,]\d+)?\b')

def extract_numbers(text: str) -> set:
    if not isinstance(text, str): return set()
    return set(NUM_REGEX.findall(text))

def preprocess(example, tokenizer, max_length):
    enc = tokenizer(
        example["context"],
        example["evidence_aug"],
        truncation=True,
        max_length=max_length,
        return_token_type_ids=True
    )

    ctx_nums = extract_numbers(example["context"])
    evid_nums = extract_numbers(example["evidence_aug"])

    has_num_mismatch = int(ctx_nums != evid_nums)
    shared_num_ratio = len(ctx_nums & evid_nums) / max(len(ctx_nums | evid_nums), 1)

    enc["num_feats"] = [has_num_mismatch, shared_num_ratio]
    enc["labels"] = example["labels"]
    return enc