import hashlib, json, sys
from datasets import load_dataset
kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
clean = json.load(open(sys.argv[1]))["clean_index"][:100]
h = hashlib.sha256()
for i in clean:
    h.update((kc[i].get("question") or kc[i].get("prompt") or "").encode())
print("clean[:100] prompt sha256 =", h.hexdigest()[:32])
