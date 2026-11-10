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
import json, os, subprocess, time, torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100 or L4"
print("GPU:", torch.cuda.get_device_name(0))

from google.colab import drive
mounted = False
for attempt in range(2):
    try:
        drive.mount("/content/drive", force_remount=(attempt > 0))
        mounted = True
        break
    except Exception as exc:
        print(f"Drive mount attempt {attempt + 1} failed: {exc}")
        time.sleep(5)

if mounted:
    BASE = "/content/drive/MyDrive/rhen"
else:
    BASE = "/content/rhen_local"
    print("WARNING: Google Drive did not mount. The corpus and checkpoints will be lost if the runtime disconnects.")

def sh(cmd, cwd="/content/RHEN"):
    """Run a shell command in the repo (or in the given directory), stream its output, and raise if it fails."""
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd}")

DRIVE_DATA = f"{BASE}/data/corpus_decision_1b"      # saved tokens (survive disconnects when Drive is mounted)
DRIVE_RUN = f"{BASE}/runs/colab_decision_1p5ep"     # checkpoints for this run (survive disconnects when Drive is mounted)
LOCAL_DATA = "/content/RHEN/data/corpus_decision_1b"  # fast local copy used for training

# The clone deletes /content/RHEN, so run it from /content.
sh("rm -rf /content/RHEN && git clone -q -b colab-lm-synth-1b https://github.com/Kivitas/RHEN.git /content/RHEN", cwd="/content")
os.chdir("/content/RHEN")
os.environ["PYTHONPATH"] = "/content/RHEN"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sh('pip install -q "transformers>=4.40,<6" "numpy>=1.24"')

if os.path.exists(f"{DRIVE_DATA}/generator.json"):
    print("Found the decision corpus in storage; copying it to local disk.")
    sh(f"mkdir -p {LOCAL_DATA} && cp {DRIVE_DATA}/train.bin {DRIVE_DATA}/val.bin {DRIVE_DATA}/meta.json {LOCAL_DATA}/ && cp {DRIVE_DATA}/generator.json {LOCAL_DATA}/")
else:
    print("No decision corpus in storage; building it (about 15-30 minutes).")
    sh(f"python tools/make_decision_corpus.py --out {LOCAL_DATA}")
    sh(f"mkdir -p {DRIVE_DATA} && cp {LOCAL_DATA}/train.bin {LOCAL_DATA}/val.bin {LOCAL_DATA}/meta.json {DRIVE_DATA}/ && cp {LOCAL_DATA}/generator.json {DRIVE_DATA}/")

sh(f"python tools/make_decision_corpus.py --out {LOCAL_DATA}")  # checks the local copy; skips if complete

# 1.5 epochs of the training split, counted in tokens: 32 sequences * 1024 tokens per step.
with open(f"{LOCAL_DATA}/meta.json", encoding="utf-8") as fh:
    TRAIN_TOKENS = json.load(fh)["train_tokens"]
TOTAL_STEPS = round(1.5 * TRAIN_TOKENS / (32 * 1024))
print(f"train tokens: {TRAIN_TOKENS:,}  ->  1.5 epochs = {TOTAL_STEPS:,} steps")

# Training. Checkpoints go to storage, so re-running this cell after a disconnect resumes.
sh(f"python -m cli.train --config configs/colab_lm_synth_1b.json --set lm_data={LOCAL_DATA} "
   f"--set run_dir={DRIVE_RUN} --set max_steps={TOTAL_STEPS} --set save_every=1000 --set eval_every=1000")
```

Settings in `configs/colab_lm_synth_1b.json`: `small` preset, `seq_len` 1024, effective batch
32 sequences per step, 2000 steps, Muon optimizer, WSD schedule. The data is random Zipf token
ids, not natural text, so the loss shows that training runs but says nothing about language quality.
