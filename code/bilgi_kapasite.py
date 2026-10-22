# -*- coding: utf-8 -*-
"""F1K — bilgi kapasitesi (Devrim, 12 Ekim 2026; ön kayıt ON_KAYIT_MERDIVEN_2026-10-08.md Ek 7 madde 7 + Ek 7b).
Allen-Zhu & Li (Physics of LMs 3.1/3.3) tarzı sentetik biyografi: N kişi, 6 özellik (doğum tarihi, doğum şehri, üniversite,
bölüm, şirket, şirket şehri). Uydurma kelimeler (gerçek dünya bilgisi yok), kelime düzeyi küçük sözlük.
Koşullar: tek (tek kalıp, sabit sıra) · cok5 (özellik başına 5 ifade + cümle sırası her görülmede karışık); her kişinin biyografisi
100 kez görülür. Kişilerin yarısının (QA-eğitim yarısı) soru-cevapları eğitimde; soru-cevap doğruluğu öbür yarıda (çıkarım ölçüsü).
Kollar: taban (AdamW, kosinüs) · paket (canon+muonw+qknorm+bigram hash+wsd). Tek boy (4×192, ≈1,8M gövde).
Ölçü: öğrenilen bit = Σ_kişi,özellik max(0, log2|havuz| − CE_bit(değer | kalıp 0 öneki)); bit/parametre (gövde ve toplam).
Güvenli duraklat: KUYRUK6_DUR (adım sonunda checkpoint, kod 3). Kullanım: python bilgi_kapasite.py --hepsi | --duman"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BURA = os.path.dirname(os.path.abspath(__file__))
MER = os.path.join(os.path.dirname(BURA), "d1_frank", "merdiven")
sys.path.insert(0, MER)
import merdiven as M  # noqa: E402

N_KISI = int(os.environ.get("BILGI_N", "50000"))
GORULME = int(os.environ.get("BILGI_GORULME", "100"))
L_, D_ = 4, 192
T = 256
B = 64
SONUC = os.path.join(BURA, "bilgi_sonuc.jsonl")
DURUM = os.path.join(BURA, "bilgi_durum.txt")
CKPT = os.path.join(BURA, "ckpt")
DUR = os.environ.get("KUYRUK6_DUR")
CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEV = "cuda" if CUDA else "cpu"
AYLAR = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
OZ = ["tarih", "sehir", "univ", "bolum", "sirket", "ssehir"]


def log(m):
    s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    try:
        print(s, flush=True)
    except Exception:  # noqa: BLE001
        pass
    with open(DURUM, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


def havuzlar():
    """Uydurma, birbirinden ayrık kelime havuzları (sabit tohum)."""
    r = random.Random(20261012)
    hece = [a + b for a in "bcdfgklmnprstvz" for b in "aeiou"]
    gor = set(AYLAR)

    def kelime(n_hece):
        while True:
            w = "".join(r.choice(hece) for _ in range(n_hece))
            if w not in gor:
                gor.add(w); return w
    H = {"ad": [kelime(2) for _ in range(400)], "soyad": [kelime(3) for _ in range(400)],
         "sehir": [kelime(3) for _ in range(200)], "univ": [kelime(3) + "u" for _ in range(300)],
         "bolum": [kelime(2) + "ics" for _ in range(100)], "sirket": [kelime(2) + "co" for _ in range(263)]}
    return H


BIT = {"tarih": math.log2(12 * 28 * 100), "sehir": math.log2(200), "univ": math.log2(300), "bolum": math.log2(100),
       "sirket": math.log2(263), "ssehir": math.log2(200)}

KALIP = {  # özellik → 5 ifade; {n} ad soyad, {v} değer (tarih: "ay gün , yıl"); ilk ifade = tek kalıp ve ölçü öneki
    "tarih": ["{n} was born on {v} .", "{n} came to this world on {v} .", "the birthday of {n} is {v} .",
              "{n} celebrates a birthday on {v} .", "on {v} , {n} was born ."],
    "sehir": ["{n} was born in {v} .", "{n} spent early years in {v} .", "the birthplace of {n} is {v} .",
              "{n} grew up in {v} .", "{n} originally comes from {v} ."],
    "univ": ["{n} studied at {v} .", "{n} graduated from {v} .", "{n} attended {v} .",
             "the alma mater of {n} is {v} .", "{n} received education at {v} ."],
    "bolum": ["{n} majored in {v} .", "{n} focused on {v} .", "the field of {n} was {v} .",
              "{n} specialized in {v} .", "{n} completed a degree in {v} ."],
    "sirket": ["{n} worked for {v} .", "{n} was employed by {v} .", "{n} built a career at {v} .",
               "the employer of {n} was {v} .", "{n} held a job at {v} ."],
    "ssehir": ["{n} worked in {v} .", "{n} was based in {v} .", "the workplace of {n} was in {v} .",
               "{n} commuted to an office in {v} .", "{n} lived and worked in {v} ."],
}
SORU = {"tarih": "q : when was {n} born ? a : {v} .", "sehir": "q : where was {n} born ? a : {v} .",
        "univ": "q : which university did {n} attend ? a : {v} .", "bolum": "q : what did {n} study ? a : {v} .",
        "sirket": "q : which company did {n} work for ? a : {v} .", "ssehir": "q : where did {n} work ? a : {v} ."}


def kisiler(H):
    r = random.Random(7)
    ciftler = r.sample([(a, s) for a in H["ad"] for s in H["soyad"]], N_KISI)
    K = []
    for a, s in ciftler:
        K.append({"n": f"{a} {s}", "tarih": f"{r.choice(AYLAR)} {r.randint(1, 28)} , {r.randint(1900, 1999)}",
                  "sehir": r.choice(H["sehir"]), "univ": r.choice(H["univ"]), "bolum": r.choice(H["bolum"]),
                  "sirket": r.choice(H["sirket"]), "ssehir": r.choice(H["sehir"])})
    return K


def sozluk(H):
    w = set(["<pad>", "<eod>"])
    for v in H.values():
        w.update(v)
    w.update(AYLAR); w.update(str(i) for i in range(1, 29)); w.update(str(y) for y in range(1900, 2000))
    for k in list(KALIP.values()) + [[q] for q in SORU.values()]:
        for t in k:
            w.update(x for x in t.replace("{n}", "").replace("{v}", "").split())
    W = sorted(w); W.remove("<pad>"); W.remove("<eod>")
    W = ["<pad>", "<eod>"] + W
    return W, {x: i for i, x in enumerate(W)}


def biyo(k, kip, r):
    if kip == "tek":
        cum = [KALIP[o][0] for o in OZ]
    else:
        cum = [r.choice(KALIP[o]) for o in OZ]; r.shuffle(cum)
    return " ".join(c.replace("{n}", k["n"]).replace("{v}", k[o]) for c, o in zip(cum, OZ if kip == "tek" else
                                                                                   [_oz_of(c) for c in cum]))


def _oz_of(c):
    for o, v in KALIP.items():
        if c in v:
            return o
    raise KeyError(c)


def tur_verisi(K, kip, tur, S2I):
    """Bir tur (bütün kişilerin biyografisi bir kez + QA yarısının soruları) — karışık sırada, <eod> ayraçlı jeton dizisi."""
    r = random.Random(1000 + tur)
    belgeler = [biyo(k, kip, r) for k in K]
    yarim = len(K) // 2
    for k in K[:yarim]:
        for o in OZ:
            belgeler.append(SORU[o].replace("{n}", k["n"]).replace("{v}", k[o]))
    r.shuffle(belgeler)
    ids = []
    for b in belgeler:
        ids.extend(S2I[x] for x in b.split()); ids.append(1)
    return np.array(ids, dtype=np.int32)


class BilgiModel(nn.Module):
    def __init__(self, V, kol):
        super().__init__()
        pk = kol == "paket"
        self.emb = nn.Embedding(V, D_)
        self.bloklar = nn.ModuleList([M.Blok(D_, max(1, D_ // 64), canon=pk, qknorm=pk) for _ in range(L_)])
        self.nf = M.RMSNorm(D_); self.bas = nn.Linear(D_, V, bias=False)
        self.bg = nn.Embedding(131071, D_) if pk else None
        self.V = V
        hd = D_ // max(1, D_ // 64)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(T + 8).float(), inv)
        self.register_buffer("cos", fr.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", fr.sin()[None, :, None, :], persistent=False)

    def forward(self, x):
        Tn = x.size(1); cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        h = self.emb(x)
        if self.bg is not None:
            h = h + self.bg((F.pad(x[:, :-1], (1, 0)) * self.V + x) % self.bg.num_embeddings)
        for b in self.bloklar:
            h = b(h, cos, sin)
        return self.bas(self.nf(h))


@torch.no_grad()
def bit_olc(model, K, S2I):
    """Kalıp 0 öneki ile değer jetonlarının çapraz entropisi (bit); öğrenilen bit = Σ max(0, log2|havuz| − CE)."""
    model.eval(); toplam = 0.0; ozel = {o: 0.0 for o in OZ}
    for o in OZ:
        on, son_ = KALIP[o][0].split("{v}")
        satirlar, hedef = [], []
        for k in K:
            a = [S2I[x] for x in on.replace("{n}", k["n"]).split()]
            v = [S2I[x] for x in k[o].split() if x != ","]
            vt = [S2I[x] for x in k[o].split()]
            satirlar.append(a + vt); hedef.append((len(a), [i for i, x in enumerate(k[o].split()) if x != ","]))
        for p0 in range(0, len(satirlar), 4096):
            grp = satirlar[p0:p0 + 4096]; hg = hedef[p0:p0 + 4096]
            Lm = max(map(len, grp))
            x = torch.zeros(len(grp), Lm, dtype=torch.long)
            for i, s in enumerate(grp):
                x[i, :len(s)] = torch.tensor(s)
            x = x.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
                lo = model(x).float()
            lp = F.log_softmax(lo, -1)
            for i, (n_on, konum) in enumerate(hg):
                ce = 0.0
                for j in konum:
                    t = n_on + j  # bu jetonu t−1 konumu tahmin eder
                    ce -= lp[i, t - 1, x[i, t]].item()
                b = max(0.0, BIT[o] - ce / math.log(2))
                ozel[o] += b; toplam += b
    model.train()
    return toplam, ozel


@torch.no_grad()
def qa_olc(model, K, S2I, I2S):
    """QA-test yarısında (eğitimde sorusu olmayan kişiler) açgözlü cevap, tam eşleşme."""
    model.eval(); dogru = {o: 0 for o in OZ}; test = K[len(K) // 2:]
    for o in OZ:
        on = SORU[o].split("{v}")[0]
        istem = [[S2I[x] for x in on.replace("{n}", k["n"]).split()] for k in test]
        gercek = [k[o] for k in test]
        for p0 in range(0, len(istem), 2048):
            grp = istem[p0:p0 + 2048]
            uz = {}
            for i, s in enumerate(grp):
                uz.setdefault(len(s), []).append(i)
            for L, ii in uz.items():
                x = torch.tensor([grp[i] for i in ii], device=DEV)
                out = []
                for _ in range(6):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
                        nx = model(x)[:, -1].float().argmax(-1)
                    out.append(nx); x = torch.cat([x, nx[:, None]], 1)
                U = torch.stack(out, 1).tolist()
                for r_, i in enumerate(ii):
                    ws = []
                    for t in U[r_]:
                        w = I2S[t]
                        if w in (".", "<eod>"):
                            break
                        ws.append(w)
                    dogru[o] += " ".join(ws) == gercek[p0 + i]
    model.train()
    return {o: round(dogru[o] / len(test), 4) for o in OZ}


def egit(kip, kol, K, S2I, I2S, tohum=1, duman=False):
    torch.manual_seed(tohum)
    V = len(S2I)
    model = BilgiModel(V, kol).to(DEV)
    M.init_(model, "gpt", 1.0, L_)
    ornek_tur = tur_verisi(K, kip, 0, S2I)
    tur_jeton = len(ornek_tur)
    adim = (tur_jeton * GORULME) // (B * T)
    if duman:
        adim = 20
    lr_tepe = 1.5e-3 * 256 / D_
    isinma = max(20, int(0.02 * adim))
    optlar = M.opt_kur(model, "muonw" if kol == "paket" else "taban", lr_tepe)

    def lr_at(s):
        if s < isinma:
            return lr_tepe * (s + 1) / isinma
        if kol == "paket":
            sg = max(1, int(0.45 * adim))
            return lr_tepe if s < adim - sg else lr_tepe * (1 - 0.9 * (s - (adim - sg)) / sg)
        p = (s - isinma) / max(1, adim - isinma)
        return lr_tepe * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    os.makedirs(CKPT, exist_ok=True)
    ck = os.path.join(CKPT, f"{kip}_{kol}_t{tohum}_{N_KISI}_{GORULME}.pt")
    s0, gecen = 0, 0.0
    if os.path.exists(ck) and not duman:
        k = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(k["model"])
        for (o, _), st in zip(optlar, k["optlar"]):
            o.load_state_dict(st)
        s0, gecen = k["s"], k["gecen"]
        log(f"{kip} {kol}: DEVAM adım {s0}/{adim}")
    # jeton akışı: tur tur üretilir (tur tohumu = tur numarası); adım s → akıştaki [s·B·T, (s+1)·B·T + 1)
    akis = {"tur": -1, "veri": None}

    def parca(bas, n):
        out = []
        while n > 0:
            tur, ofs = divmod(bas, tur_jeton)
            if akis["tur"] != tur:
                akis["tur"], akis["veri"] = tur, tur_verisi(K, kip, tur, S2I)
                if len(akis["veri"]) < tur_jeton:  # turlar arasında uzunluk farkı (cok5) → dolgu
                    akis["veri"] = np.pad(akis["veri"], (0, tur_jeton - len(akis["veri"])), constant_values=1)
            al = min(n, tur_jeton - ofs)
            out.append(akis["veri"][ofs:ofs + al]); bas += al; n -= al
        return np.concatenate(out)

    t0 = time.time(); son = time.time(); iz = []
    for s in range(s0, adim):
        for o, b0 in optlar:
            for g in o.param_groups:
                g["lr"] = lr_at(s) if b0 == lr_tepe else b0 * lr_at(s) / lr_tepe
        a = torch.from_numpy(parca(s * B * T, B * T + 1).astype(np.int64))
        x, y = a[:-1].view(B, T).to(DEV), a[1:].view(B, T).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
            lo = model(x)
        loss = F.cross_entropy(lo.float().view(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, _ in optlar:
            o.step(); o.zero_grad(set_to_none=True)
        if s % max(1, adim // 10) == 0 or s == adim - 1:
            iz.append((s, round(loss.item(), 4)))
            print(f"[{kip}/{kol}] adım {s}/{adim} kayıp {loss.item():.4f}", flush=True)
        if not duman and s < adim - 1 and DUR and (os.path.exists(DUR) or time.time() - son > 600):
            torch.save(dict(model=model.state_dict(), optlar=[o.state_dict() for o, _ in optlar], s=s + 1, gecen=gecen + time.time() - t0),
                       ck + ".tmp")
            os.replace(ck + ".tmp", ck); son = time.time()
            if os.path.exists(DUR):
                log(f"{kip} {kol}: DURAKLATILDI adım {s + 1}/{adim}")
                sys.exit(3)
    sure = gecen + time.time() - t0
    bit, ozel = bit_olc(model, K, S2I)
    qa = qa_olc(model, K, S2I, I2S)
    gov = sum(p.numel() for n, p in model.named_parameters() if n.startswith("bloklar"))
    top = sum(p.numel() for p in model.parameters())
    tablo = sum(p.numel() for n, p in model.named_parameters() if n.startswith("bg"))
    satir = dict(kip=kip, kol=kol, tohum=tohum, N_kisi=N_KISI, gorulme=GORULME, adim=adim, jeton=adim * B * T, egitim_sn=round(sure, 1),
                 bilgi_bit_toplam=round(sum(BIT.values()) * N_KISI), ogrenilen_bit=round(bit), ozellik_bit={o: round(v) for o, v in ozel.items()},
                 N_govde=gov, N_toplam=top, N_tablo=tablo, bit_param_govde=round(bit / gov, 4), bit_param_toplam=round(bit / top, 4),
                 bit_param_tablosuz=round(bit / (top - tablo), 4), qa_test=qa, qa_ort=round(sum(qa.values()) / len(qa), 4), kayip_iz=iz,
                 zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
    return satir


def rapor(R):
    sat = ["## F1K — bilgi kapasitesi (Ek 7 madde 7, Ek 7b)", "",
           f"Sentetik biyografi: {N_KISI:,} kişi × 6 özellik ≈ {sum(BIT.values()):.1f} bit/kişi (toplam {sum(BIT.values()) * N_KISI / 1e6:.2f}M bit); "
           f"her biyografi {GORULME} kez. QA doğruluğu: eğitimde sorusu OLMAYAN yarıda.", "",
           "| Koşul | Kol | Öğrenilen bit | bit/param (gövde) | bit/param (tablo hariç toplam) | bit/param (toplam) | QA ort (test yarısı) |",
           "|---|---|---|---|---|---|---|"]
    for x in R:
        sat.append(f"| {x['kip']} | {x['kol']} | {x['ogrenilen_bit'] / 1e6:.3f}M | {x['bit_param_govde']:.3f} | {x['bit_param_tablosuz']:.3f} | "
                   f"{x['bit_param_toplam']:.3f} | {x['qa_ort']:.3f} |")
    hk = []
    bul = lambda kip, kol: next((x for x in R[::-1] if x["kip"] == kip and x["kol"] == kol), None)  # noqa: E731
    for kol in ("taban", "paket"):
        a, b = bul("tek", kol), bul("cok5", kol)
        if a and b:
            fark = 100 * (b["qa_ort"] - a["qa_ort"])
            hk.append(f"- {kol}: ifade çeşitliliği (cok5 − tek) QA {fark:+.1f} puan ⇒ "
                      + ("veri yoğunlaştırma yararlı" if fark >= 5 else "kazanç < 5 puan: veri yoğunlaştırmaya yatırım azalır (K1K)"))
    for x in R:
        if x["bit_param_tablosuz"] > 3:
            hk.append(f"- {x['kip']} {x['kol']}: tablo hariç {x['bit_param_tablosuz']:.2f} bit/param > 3 ⇒ OLAĞANÜSTÜ İŞARET (tohum 2 gerekli)")
    hk.append("- Literatür: ≈2 bit/param ≈1000 görülmede, ≈1 bit/param 100 görülmede (Allen-Zhu & Li 2024, sentetik veri).")
    sat += ["", *hk]
    open(os.path.join(MER, "kuyruk6_rapor_F1K.md"), "w", encoding="utf-8", newline="\n").write("\n".join(sat) + "\n")
    return hk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hepsi", action="store_true"); ap.add_argument("--duman", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    H = havuzlar(); K = kisiler(H); W, S2I = sozluk(H); I2S = {i: w for w, i in S2I.items()}
    log(f"F1K: {len(K):,} kişi, sözlük {len(W)} kelime, kişi başına {sum(BIT.values()):.1f} bit")
    R = [json.loads(l) for l in open(SONUC, encoding="utf-8")] if os.path.exists(SONUC) else []
    for kip in ("tek", "cok5"):
        for kol in ("taban", "paket"):
            if a.duman and (kip, kol) != ("cok5", "paket") and (kip, kol) != ("tek", "taban"):
                continue
            if not a.duman and any(x["kip"] == kip and x["kol"] == kol and x["N_kisi"] == N_KISI and x["gorulme"] == GORULME for x in R):
                log(f"zaten var: {kip} {kol}"); continue
            log(f"BASLADI {kip} {kol}")
            s = egit(kip, kol, K, S2I, I2S, duman=a.duman)
            log(f"BITTI {kip} {kol}: {s['ogrenilen_bit'] / 1e6:.3f}M bit, {s['bit_param_govde']:.3f} bit/param gövde, QA {s['qa_ort']:.3f} "
                f"({s['egitim_sn'] / 60:.1f} dk)")
            if not a.duman:
                with open(SONUC, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                R.append(s)
                ck = os.path.join(CKPT, f"{kip}_{kol}_t1_{N_KISI}_{GORULME}.pt")
                if os.path.exists(ck):
                    os.remove(ck)
            if CUDA:
                torch.cuda.empty_cache()
    if a.duman:
        log("DUMAN BITTI"); return
    hk = rapor(R)
    log("F1K BITTI: " + " | ".join(hk))


if __name__ == "__main__":
    main()
