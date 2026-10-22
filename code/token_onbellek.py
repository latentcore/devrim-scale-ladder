# -*- coding: utf-8 -*-
"""Merdiven (Aşama 4 aleti) için token önbelleği: ClimbMix eğitim parçaları (prepare.py sırası, val hariç)
ve sabit doğrulama parçası, aynı 8192 BPE ile bir kez tokenize edilip uint16 dosyalara yazılır.
Her belge BOS ile başlar (make_dataloader ile aynı). Belge başlangıçları ayrı .npy'de (müfredat için).
Kullanım: python token_onbellek.py [--train-tokens 400000000] [--val-tokens 4000000]"""
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq

AR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "autoresearch")
sys.path.insert(0, os.path.abspath(AR))
from prepare import Tokenizer, list_parquet_files, DATA_DIR, VAL_FILENAME, CACHE_DIR  # noqa: E402

OUT = os.path.join(CACHE_DIR, "merdiven")


def yaz(paths, limit, ad, tok, bos):
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, ad + ".bin.tmp")
    starts, n = [], 0
    t0 = time.time()
    with open(tmp, "wb") as fh:
        for p in paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                texts = pf.read_row_group(rg).column("text").to_pylist()
                for i in range(0, len(texts), 512):
                    ids = tok.encode(texts[i:i + 512], prepend=bos, num_threads=8)
                    for row in ids:
                        if n >= limit:
                            break
                        row = row[: limit - n]
                        starts.append(n)
                        fh.write(np.asarray(row, dtype=np.uint16).tobytes())
                        n += len(row)
                    if n >= limit:
                        break
                print(f"  {ad}: {os.path.basename(p)} rg {rg + 1}/{pf.num_row_groups} -> {n / 1e6:.1f}M token ({time.time() - t0:.0f} sn)", flush=True)
                if n >= limit:
                    break
            if n >= limit:
                break
    os.replace(tmp, os.path.join(OUT, ad + ".bin"))
    np.save(os.path.join(OUT, ad + "_doc_starts.npy"), np.asarray(starts, dtype=np.int64))
    print(f"{ad}: {n:,} token, {len(starts):,} belge, {time.time() - t0:.0f} sn", flush=True)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=400_000_000)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    a = ap.parse_args()
    tok = Tokenizer.from_directory()
    assert tok.get_vocab_size() <= 65535
    bos = tok.get_bos_token_id()
    files = list_parquet_files()
    val = os.path.join(DATA_DIR, VAL_FILENAME)
    train = [p for p in files if p != val]
    print("egitim parcalari:", [os.path.basename(p) for p in train])
    yaz([val], a.val_tokens, "val", tok, bos)
    yaz(train, a.train_tokens, "train", tok, bos)
    print("BITTI", OUT)
