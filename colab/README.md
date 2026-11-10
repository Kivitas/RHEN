# Train RHEN on Google Colab (1B-token synthetic corpus)

The corpus is not committed: `train.bin` is about 2 GB and GitHub rejects files over 100 MB.
Instead, the cell below rebuilds it. `tools/make_synthetic_corpus.py` is deterministic, so every
run produces the same bytes.

**Runtime:** Runtime → Change runtime type → Python 3, **A100 GPU** (best), or **L4 GPU**.
The **T4** also works, but it has no bf16 support. If you use a T4, change `"precision": "auto"`
to `"fp16"` in `configs/colab_lm_synth_1b.json`.
**High RAM:** not needed. The corpus is memory-mapped and the `small` model is tiny.

## Paste this into one Colab cell

```python
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100 or L4"
print("GPU:", torch.cuda.get_device_name(0))

!rm -rf /content/RHEN && git clone -q -b colab-lm-synth-1b https://github.com/Kivitas/RHEN.git /content/RHEN
%cd /content/RHEN
%env PYTHONPATH=/content/RHEN
!pip install -q "transformers>=4.40,<6" "numpy>=1.24"

# Build the 1B-token corpus (about 3-5 minutes, about 2 GB on disk).
!python tools/make_synthetic_corpus.py --out data/corpus_synth_1b

# Train. Checkpoints go to runs/colab_lm_synth_1b. Re-running this cell resumes from the latest one.
!python -m cli.train --config configs/colab_lm_synth_1b.json
```

Settings in `configs/colab_lm_synth_1b.json`: `small` preset, `seq_len` 1024, effective batch
32 sequences per step, 2000 steps, Muon optimizer, WSD schedule. The data is random Zipf token
ids, not natural text, so the loss shows that training runs but says nothing about language quality.
