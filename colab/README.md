# Train RHEN on Google Colab (1B-token synthetic corpus)

The corpus is not committed: `train.bin` is about 2 GB and GitHub rejects files over 100 MB.
The cell below builds it once and saves it to Google Drive. After a disconnect it copies the
saved `.bin` files back instead of rebuilding them. Checkpoints also go to Drive, so re-running
the cell resumes training from the last checkpoint.

**Runtime:** Runtime → Change runtime type → Python 3, **A100 GPU** (best), or **L4 GPU**.
The **T4** also works, but it has no bf16 support. If you use a T4, change `"precision": "auto"`
to `"fp16"` in `configs/colab_lm_synth_1b.json`.
**High RAM:** not needed. The corpus is memory-mapped and the `small` model is tiny.

## Paste this into one Colab cell

```python
import os, subprocess, torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100 or L4"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount("/content/drive")  # the first run asks you to sign in once

def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)

DRIVE_DATA = "/content/drive/MyDrive/rhen/data/corpus_synth_1b"    # saved tokens (survive disconnects)
DRIVE_RUN = "/content/drive/MyDrive/rhen/runs/colab_lm_synth_1b"   # checkpoints (survive disconnects)
LOCAL_DATA = "/content/RHEN/data/corpus_synth_1b"                  # fast local copy used for training

sh("rm -rf /content/RHEN && git clone -q -b colab-lm-synth-1b https://github.com/Kivitas/RHEN.git /content/RHEN")
os.chdir("/content/RHEN")
os.environ["PYTHONPATH"] = "/content/RHEN"
sh('pip install -q "transformers>=4.40,<6" "numpy>=1.24"')

if os.path.exists(f"{DRIVE_DATA}/generator.json"):
    print("Found the corpus on Drive; copying it to local disk.")
    sh(f"mkdir -p {LOCAL_DATA} && cp {DRIVE_DATA}/train.bin {DRIVE_DATA}/val.bin {DRIVE_DATA}/meta.json {LOCAL_DATA}/ && cp {DRIVE_DATA}/generator.json {LOCAL_DATA}/")
else:
    print("No corpus on Drive; building it (about 3-5 minutes).")
    sh(f"python tools/make_synthetic_corpus.py --out {LOCAL_DATA}")
    sh(f"mkdir -p {DRIVE_DATA} && cp {LOCAL_DATA}/train.bin {LOCAL_DATA}/val.bin {LOCAL_DATA}/meta.json {DRIVE_DATA}/ && cp {LOCAL_DATA}/generator.json {DRIVE_DATA}/")

sh(f"python tools/make_synthetic_corpus.py --out {LOCAL_DATA}")  # checks the local copy; skips if complete

# Training. Checkpoints go to Drive, so re-running this cell after a disconnect resumes.
sh(f"python -m cli.train --config configs/colab_lm_synth_1b.json --set lm_data={LOCAL_DATA} --set run_dir={DRIVE_RUN}")
```

Settings in `configs/colab_lm_synth_1b.json`: `small` preset, `seq_len` 1024, effective batch
32 sequences per step, 2000 steps, Muon optimizer, WSD schedule. The data is random Zipf token
ids, not natural text, so the loss shows that training runs but says nothing about language quality.
