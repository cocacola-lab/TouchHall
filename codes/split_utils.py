from pathlib import Path

import torch


def metadata_key(row):
    source_file = row.get("source_file")
    source_index = row.get("source_index")
    if source_file is None or source_index is None:
        raise ValueError("Each sample must contain source_file and source_index for split alignment.")
    return Path(str(source_file)).name, int(source_index)


def metadata_keys(metadata):
    keys = [metadata_key(row) for row in metadata]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (source_file, source_index) identities found in metadata.")
    return keys


def load_aligned_split(split_file, metadata):
    split = torch.load(Path(split_file).expanduser(), map_location="cpu")
    current_keys = metadata_keys(metadata)
    current_index = {key: index for index, key in enumerate(current_keys)}

    train_keys = [tuple(key) for key in split["train_keys"]]
    val_keys = [tuple(key) for key in split["val_keys"]]
    split_keys = train_keys + val_keys

    missing = [key for key in split_keys if key not in current_index]
    extra = [key for key in current_keys if key not in set(split_keys)]
    if missing or extra:
        raise ValueError(
            "Training data does not match the shared split: "
            f"missing={missing[:5]} (total {len(missing)}), "
            f"extra={extra[:5]} (total {len(extra)})."
        )

    train_idx = torch.tensor([current_index[key] for key in train_keys], dtype=torch.long)
    val_idx = torch.tensor([current_index[key] for key in val_keys], dtype=torch.long)
    return train_idx, val_idx, split
