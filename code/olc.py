# -*- coding: utf-8 -*-
"""Değerlendirme (00_PLAN.md madde 6): egit_evren.py kaydını açık testlerde ölçer — HellaSwag (val), ARC-Easy, ARC-Challenge,
MMLU (test; sıfır atış, cloze), TurkishMMLU (900 soru, 5 şık). Yöntem lm-eval ile aynı ilke: her seçeneğin bağlama göre log-olasılığı; acc = en yüksek toplam,
acc_norm = bayt başına (HellaSwag/ARC için yaygın). Veri: olcum_veri\\*.parquet (Hugging Face, 8 Ekim indirildi).
Çıta: Haiku 4.5'in bu testlerdeki puanı yayımlanmadı; karşılaştırma için açık modellerin yayımlanmış puanları kullanılacak (DOĞRULANMADI).
Kullanım: python olc.py --kayit <kayit_dizini> [--cihaz cuda] [--limit 1000]  → <kayit>\\olcum.json"""
import argparse, ast, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
sys.path.insert(0, os.path.join(BURA, "..", "d1_frank", "autoresearch"))
import egit_evren as E  # noqa: E402
from prepare import Tokenizer  # noqa: E402

VERI = os.path.join(BURA, "olcum_veri")
SANS = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "mmlu": 0.25, "turkishmmlu": 0.2}


def liste(v):
    return list(v) if not isinstance(v, str) else ast.literal_eval(v)


def gorevler(limit):
    g = {}
    t = pq.read_table(os.path.join(VERI, "hellaswag_val.parquet")).to_pylist()[:limit]
    g["hellaswag"] = [(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize(),
                       [" " + e.replace(" [title]", ". ").strip() for e in liste(r["endings"])], int(r["label"])) for r in t]
    for ad in ("arc_easy", "arc_challenge"):
        t = pq.read_table(os.path.join(VERI, f"{ad}_test.parquet")).to_pylist()[:limit]
        g[ad] = [("Question: " + r["question"] + "\nAnswer:", [" " + c for c in r["choices"]["text"]],
                  list(r["choices"]["label"]).index(r["answerKey"])) for r in t]
    t = pq.read_table(os.path.join(VERI, "mmlu_test.parquet")).to_pylist()
    rng = np.random.default_rng(0); sec = rng.permutation(len(t))[:limit]  # konulara dağılsın diye karışık alt küme (sabit tohum)
    g["mmlu"] = [("Question: " + t[i]["question"] + "\nAnswer:", [" " + str(c) for c in liste(t[i]["choices"])], int(t[i]["answer"])) for i in sec]
    # TurkishMMLU (900 soru, 9 ders, 5 şık; MertCore DIS-1'den kopyalandı, 8 Ekim) — Türkçe ölçü
    tq = json.load(open(os.path.join(VERI, "turkishmmlu_sorular.json"), encoding="utf-8"))
    ta = json.load(open(os.path.join(VERI, "turkishmmlu_anahtar.json"), encoding="utf-8"))
    g["turkishmmlu"] = [("Soru: " + q["question"] + "\nCevap:", [" " + str(c) for c in liste(q["choices"])], "ABCDE".index(ta[q["id"]]))
                        for q in tq[:limit] if q["id"] in ta]
    return g


@torch.no_grad()
def puanla(model, tok, ogeler, T, cihaz):
    dogru = dogru_n = 0
    bos = tok.get_bos_token_id()
    for baglam, secenekler, altin in ogeler:
        b_ids = [bos] + tok.encode(baglam)
        lp, nb = [], []
        for s in secenekler:
            s_ids = tok.encode(s)
            ids = (b_ids + s_ids)[-(T + 1):]
            x = torch.tensor(ids[:-1], device=cihaz)[None]; y = torch.tensor(ids[1:], device=cihaz)
            with torch.autocast(cihaz.type, dtype=torch.bfloat16, enabled=(cihaz.type == "cuda")):
                lo, _ = model(x)
            l = F.log_softmax(lo[0].float(), -1).gather(1, y[:, None])[:, 0]
            lp.append(l[-len(s_ids):].sum().item()); nb.append(len(s.encode("utf-8")))
        dogru += int(np.argmax(lp) == altin); dogru_n += int(np.argmax(np.array(lp) / np.array(nb)) == altin)
    n = len(ogeler)
    return dict(n=n, acc=round(dogru / n, 4), acc_norm=round(dogru_n / n, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kayit", required=True); ap.add_argument("--cihaz", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    k = torch.load(os.path.join(a.kayit, "son.pt"), map_location="cpu", weights_only=False)
    ns = argparse.Namespace(**k["args"])
    model = E.model_kur(ns, set(ns.kol.split("+")))
    model.load_state_dict(k["model"])
    cihaz = torch.device("cuda" if a.cihaz == "cuda" else "cpu")
    model.to(cihaz).eval()
    tok = Tokenizer.from_directory()
    sonuc = dict(kayit=a.kayit, kol=ns.kol, L=ns.L, d=ns.d, adim=k["adim"], limit=a.limit, zaman=time.strftime("%Y-%m-%d %H:%M:%S"), testler={})
    for ad, ogeler in gorevler(a.limit).items():
        t0 = time.time()
        r = puanla(model, tok, ogeler, ns.T, cihaz); r["sans"] = SANS[ad]; r["sn"] = round(time.time() - t0, 1)
        sonuc["testler"][ad] = r
        print(ad, r, flush=True)
    json.dump(sonuc, open(os.path.join(a.kayit, "olcum.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
