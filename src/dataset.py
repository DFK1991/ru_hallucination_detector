from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_dataset_splits(ds_filename):

    ds = load_dataset("parquet", data_files={"train": ds_filename + ".parquet"})["train"]

    labels = ds["aug_type"]
    idx = list(range(len(ds)))

    # split
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42, stratify=labels)

    train_labels = [labels[i] for i in train_idx]

    train_idx, val_idx = train_test_split(train_idx, test_size=0.1111, random_state=42, stratify=train_labels)

    dataset = DatasetDict({"train": ds.select(train_idx), "validation": ds.select(val_idx), "test": ds.select(test_idx),})

    # label mapping
    unique_labels = sorted(set(dataset["train"]["aug_type"]))

    idx2label = {i: k for i, k in enumerate(unique_labels)}
    label2idx = {k: i for i, k in enumerate(unique_labels)}

    return dataset, idx2label, label2idx