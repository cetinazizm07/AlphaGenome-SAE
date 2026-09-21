"""Small on-disk caches with independently specified genomic rows."""
import numpy as np
import pandas as pd
from ag_sae.data import COORDINATES, atomic_json, sha256


def write_cache(root, ann, values, shard_size=5):
    root.mkdir(parents=True, exist_ok=True)
    index, cursor = [], 0
    while cursor < len(values):
        split = ann.split.iloc[cursor]
        end = cursor + 1
        while end < min(cursor + shard_size, len(values)) and ann.split.iloc[end] == split:
            end += 1
        name = f"act_{len(index):04d}.npy"
        np.save(root / name, values[cursor:end])
        index.append({"shard": name, "split": split, "row_start": cursor,
                      "row_end": end, "sha256": sha256(root / name)})
        cursor = end
    pd.DataFrame(index).to_parquet(root / "shard_index.parquet", index=False)
    ann[COORDINATES].to_parquet(root / "row_coordinates.parquet", index=False)
    atomic_json(root / "extraction_meta.json", {"format_version": 2, "dim": values.shape[1],
                "total_rows": len(values), "coordinates_sha256": sha256(root / "row_coordinates.parquet"),
                "shard_index_sha256": sha256(root / "shard_index.parquet")})
