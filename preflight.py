"""Checks that the analysis stage reproduces the training stage exactly."""
import json
import numpy as np
import torch

from ag_sae.extract import ShardStore
from ag_sae.concepts import FrozenSAE
from ag_sae import sae as S

ACTS = "/mnt/ag/smoke/acts_smoke"
CKPT = "/mnt/ag/smoke/sae_smoke/best.pt"
TAP = "resid_pre_b8"

store = ShardStore(ACTS, TAP, "val")
rows = store.take(np.arange(4096))
print("shard dtype on disk:", store.arrays[0].dtype, "| after take:", rows.dtype)

# Rebuild the training-time model from the resume checkpoint.
latest = torch.load("/mnt/ag/smoke/sae_smoke/latest.pt", map_location="cpu",
                    weights_only=False)
recipe_d = latest["identity"]["recipe"]
recipe = S.Recipe(**{k: v for k, v in recipe_d.items()
                     if k in S.Recipe.__dataclass_fields__})
model = S.build(store.dim, recipe, latest["channel_scale"], "cpu")
model.load_state_dict(latest["model"])
model.eval()

with torch.no_grad():
    _, expected, _ = model(torch.from_numpy(rows))
expected = expected.numpy()

frozen = FrozenSAE.from_torch_checkpoint(CKPT)
got = frozen.encode(rows).toarray()

print("k from recipe:", recipe.k(store.dim), "| frozen k:", frozen.k)
print("channel_scale in checkpoint:", frozen.channel_scale is not None)
print("token_layernorm:", frozen.token_layernorm)
print("nonzeros per row  torch:", (expected != 0).sum(1).mean(),
      " frozen:", (got != 0).sum(1).mean())
same_support = ((expected != 0) == (got != 0)).all()
maxdiff = float(np.abs(expected - got).max())
print("identical support:", bool(same_support))
print("max abs difference:", maxdiff)
assert same_support, "TopK picked different features"
assert maxdiff < 1e-4, f"code values disagree by {maxdiff}"
print("ROUND TRIP OK")

# best.pt must come from the same run as latest.pt
blob = torch.load(CKPT, map_location="cpu", weights_only=True)
print("best.pt step:", blob.get("step"), "| latest step:", latest["step"])

# float16 headroom on the real shards
mx = max(float(np.abs(np.asarray(a[:20000], dtype=np.float32)).max())
         for a in store.arrays)
print("max |activation| on val shards:", round(mx, 1), "| float16 max: 65504")
