"""ClimbMix parçalarını autoresearch'ün beklediği yere indirir (prepare.py'deki adlar ve URL birebir)."""
import os
import sys
import requests

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "data")
IDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 6)) + [6542]

os.makedirs(DATA_DIR, exist_ok=True)
for i in IDS:
    name = f"shard_{i:05d}.parquet"
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        print("var:", name, flush=True)
        continue
    tmp = path + ".tmp"
    for attempt in range(5):
        try:
            with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, path)
            print("indi:", name, os.path.getsize(path), flush=True)
            break
        except Exception as e:
            print("hata:", name, attempt, e, flush=True)
print("BITTI", flush=True)
