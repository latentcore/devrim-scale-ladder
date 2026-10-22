# -*- coding: utf-8 -*-
"""Etkin hesap çarpanı (m) ve "Haiku 4.5'e uzaklık" yüzdesi (ön kayıt ON_KAYIT_MERDIVEN_2026-10-08.md, Ek 3).
Ek 3a (sağlama tutmadığı için, aday sonucu görülmeden): aday AYNI boyuttaki taban ortalamasıyla (tohum 1+2) karşılaştırılır;
kayıp farkı, taban eğrisinin eğimiyle (b; g2–g4, ln L = a − b·ln C) hesaba çevrilir:
m = (C_taban,boy / C_aday) · exp((ln L_taban,boy − ln L_aday) / b) ; yüzde = 100·log10(m)/5 (hedef H = 10^5 kat: Haiku 4.5 sınıfı ≈1e24 işlem ÷ laptop haftası ≈1e19).
İkincil: duvar saati çarpanı (tabanın süre eğrisiyle). Kullanım: python katsayi.py [jsonl ...] → katsayi_rapor.md"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V = 8192
H_HEDEF = 1e5
ESIK_GERCEK, ESIK_YUZDE5 = 1.10, 10 ** 0.25


def oku(yollar):
    S = []
    for y in yollar:
        if os.path.exists(y):
            with open(y, encoding="utf-8") as fh:
                S += [json.loads(l) for l in fh if l.strip()]
    return S


def flop(s):
    """merdiven.py flop_hesap ile aynı: 6 · (gövde + çıkış başı) · token."""
    if s.get("flop"):
        return float(s["flop"])
    k = set(s["kol"].split("+"))
    bas = V * s["d"] * (2 if "mtp2" in k else 1)
    g = s["N"] * (2 if "dongu2" in k else 1)
    return 6.0 * (g + bas) * s["tokens"]


def taban_egrisi(S, boylar=("g2", "g3", "g4")):
    """(a, b, Cmin, Cmax, sa, sb): ln L = a − b ln C ; ln t = sa + sb ln C (duvar saati)."""
    nok = {}
    for s in S:
        if s["arch"] == "gpt" and s["kol"] == "taban" and s["tohum"] in (1, 2) and s["boy"] in boylar and s.get("derle") is not False:
            nok.setdefault(s["boy"], []).append((flop(s), s["val_nat"], s["egitim_sn"]))
    assert len(nok) >= 3, f"taban noktası eksik: {sorted(nok)}"
    C = np.array([np.mean([x[0] for x in v]) for v in nok.values()])
    L = np.array([np.mean([x[1] for x in v]) for v in nok.values()])
    T = np.array([np.mean([x[2] for x in v]) for v in nok.values()])
    b, a = np.polyfit(np.log(C), np.log(L), 1)
    sb, sa = np.polyfit(np.log(C), np.log(T), 1)
    return a, -b, C.min(), C.max(), sa, sb


TABAN_BOY = {}
EAGER = {"oran": None, "kayip_farki": []}


def taban_boy(S):
    """boy -> (ortalama C, ortalama L, ortalama süre) taban tohum 1+2."""
    d = {}
    for s in S:
        if s["arch"] == "gpt" and s["kol"] == "taban" and s["tohum"] in (1, 2) and s.get("derle") is not False:
            d.setdefault(s["boy"], []).append((flop(s), s["val_nat"], s["egitim_sn"]))
    out = {b: tuple(float(np.mean([x[i] for x in v])) for i in range(3)) for b, v in d.items()}
    # derlemesiz (eager) taban koşusu varsa süre oranı: eager / derlenmiş (süre çarpanı derlemesiz adaylarda bununla düzeltilir)
    oran = [s["egitim_sn"] / out[s["boy"]][2] for s in S if s["arch"] == "gpt" and s["kol"] == "taban" and s.get("derle") is False and s["boy"] in out]
    EAGER["oran"] = float(np.mean(oran)) if oran else None
    EAGER["kayip_farki"] = [round(s["val_nat"] - out[s["boy"]][1], 4) for s in S if s["arch"] == "gpt" and s["kol"] == "taban" and s.get("derle") is False and s["boy"] in out]
    return out


def carpan(s, eg):
    a, b, cmin, cmax, sa, sb = eg
    Cb, Lb, Tb = TABAN_BOY[s["boy"]]
    oran = math.exp((math.log(Lb) - math.log(s["val_nat"])) / b)  # tabanın bu kayba ulaşmak için gereken hesap katı
    m = Cb * oran / flop(s)
    if s.get("derle") is False:  # derlemesiz koşu: tabanın süresi eager oranıyla düzeltilir (oran yoksa süre kıyası yok)
        Tb = Tb * EAGER["oran"] if EAGER["oran"] else float("nan")
    m_t = Tb * oran ** sb / s["egitim_sn"]
    dis = oran > 4 or oran < 0.25
    return m, m_t, dis


def yuzde(m):
    return 100.0 * math.log10(m) / math.log10(H_HEDEF)


def tablo(S, eg, filtre=None):
    satirlar = []
    for s in S:
        if s["arch"] != "gpt" or (filtre and not filtre(s)):
            continue
        m, m_t, dis = carpan(s, eg)
        satirlar.append((s["kol"], s["boy"], s["tohum"], s.get("etiket", ""), s["val_nat"], m, yuzde(m), m_t, dis))
    return satirlar


def main():
    yollar = sys.argv[1:] or [os.path.join(HERE, "merdiven_sonuc.jsonl"), os.path.join(HERE, "muon_lr_tarama.jsonl")]
    S = oku(yollar)
    eg = taban_egrisi(S)
    TABAN_BOY.update(taban_boy(S))
    r = ["# Etkin hesap çarpanı — Haiku 4.5'e uzaklık (katsayi.py, ön kayıt Ek 3)", "",
         f"Aynı boydaki taban ortalamasıyla karşılaştırma (Ek 3a); eğim b = {eg[1]:.5f} (g2–g4, tohum 1+2). Hedef H = 10^5 kat; yüzde = 100·log10(m)/5; "
         f"m ≥ {ESIK_GERCEK:.2f} gerçek kazanç, m ≥ {ESIK_YUZDE5:.2f} = %5. m tabanımıza göredir. "
         f"Derlemesiz (eager) taban: süre oranı {EAGER['oran']}, kayıp farkı {EAGER['kayip_farki']} (matematik aynı olmalı).", "",
         "| Kol | Boy | Tohum | Etiket | Kayıp (nat) | m (hesap) | Haiku'ya % | m (süre) | Not |", "|---|---|---|---|---|---|---|---|---|"]
    for kol, boy, toh, et, L, m, p, mt, dis in sorted(tablo(S, eg), key=lambda x: (x[0] != "taban", x[0], x[1], x[2])):
        r.append(f"| {kol} | {boy} | {toh} | {et} | {L:.4f} | {m:.3f} | {p:+.2f} | {mt:.3f} | {'dış değerleme' if dis else ''} |")
    open(os.path.join(HERE, "katsayi_rapor.md"), "w", encoding="utf-8", newline="\n").write("\n".join(r) + "\n")
    print("\n".join(r))


if __name__ == "__main__":
    main()
