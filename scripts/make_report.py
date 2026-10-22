# -*- coding: utf-8 -*-
"""F5 sayilari: HAM dosyalardan (sonuc.json, gunluk.jsonl, merdiven_sonuc.jsonl, t1_taban_ayar.jsonl, akil_sonuc.jsonl, bilgi_sonuc.jsonl,
kuyruk6_durum.txt) bagimsiz yeniden hesap; kuyruk raporlariyla karsilastirma; tablolar (tables.tex), sekil, sayilar.json, UYUSMAZLIK_RAPORU.md.
Kuyruk kodunu (k6_ortak / katsayi) KULLANMAZ: formul burada ayri yazildi (m = (C_taban/C_kol) * (L_taban/L_kol)^(1/b)).
Kullanim: python hesap_ham.py   (hicbir kaynak dosyaya yazmaz; yalnizca bu klasore)."""
import json, math, os, re, sys, datetime
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "..", "results", "main_ladder_runs.jsonl")):
    # open-repo layout: <repo>/scripts/make_report.py ; inputs <repo>/results/* ; outputs <repo>/paper (tables, figure) and <repo>/results
    R = os.path.abspath(os.path.join(HERE, ".."))
    RES = os.path.join(R, "results")
    OUT = os.path.join(R, "paper")
    OUT_NUM = os.path.join(RES, "numbers.json")
    OUT_REP = os.path.join(RES, "comparison_report_tr.md")
    BIG_JSONL = os.path.join(RES, "main_ladder_runs.jsonl")
    BM = None
    P_SMALL = os.path.join(RES, "small_ladder_runs.jsonl")
    P_T1 = os.path.join(RES, "t1_baseline_tuning.jsonl")
    P_F1A = os.path.join(RES, "f1a_results.json")
    P_AK = os.path.join(RES, "f1b_reasoning_results.jsonl")
    P_BK = os.path.join(RES, "f1k_knowledge_results.jsonl")
    P_QLOG = os.path.join(RES, "queue_status_log.txt")
    SCRIPT = "scripts/make_report.py"
else:
    # yerel calisma duzeni (Devrim klasoru): bu dosya <Devrim>/F5_TASLAK icinde
    R = os.path.abspath(os.path.join(HERE, ".."))
    OUT = HERE
    OUT_NUM = os.path.join(OUT, "sayilar.json")
    OUT_REP = os.path.join(OUT, "UYUSMAZLIK_RAPORU.md")
    BIG_JSONL = None
    BM = os.path.join(R, "EVREN_TEK_ATIS", "buyuk_merdiven")
    MR = os.path.join(R, "d1_frank", "merdiven")
    P_SMALL = os.path.join(MR, "merdiven_sonuc.jsonl")
    P_T1 = os.path.join(MR, "t1_taban_ayar.jsonl")
    P_F1A = os.path.join(MR, "f1a_sonuc.json")
    P_AK = os.path.join(R, "F1B_akil_cekirdek", "akil_sonuc.jsonl")
    P_BK = os.path.join(R, "F1K_bilgi_kapasite", "bilgi_sonuc.jsonl")
    P_QLOG = os.path.join(MR, "kuyruk6_durum.txt")
    SCRIPT = "F5_TASLAK\\hesap_ham.py"
V = 8192
GOZLEM = []      # (ad, hesaplanan, raporlanan, tolerans, tamam_mi)
NOT = []         # serbest uyusmazlik / gozlem notlari


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def kars(ad, hes, rap, tol):
    ok = abs(hes - rap) <= tol
    GOZLEM.append((ad, hes, rap, tol, ok))
    return ok


def slope(C, L):
    return float(-np.polyfit(np.log(C), np.log(L), 1)[0])


def m_of(base, run, b):
    """base, run: (flop, val_nat)."""
    return (base[0] / run[0]) * (base[1] / run[1]) ** (1.0 / b)


# ------------------------------------------------------------------ buyuk merdiven (S1-S3)
def load_runs():
    """run adi -> (sonuc sozlugu ya da None, gunluk satirlari). Yerel duzen: <run>/sonuc.json + gunluk.jsonl; depo duzeni: tek main_ladder_runs.jsonl."""
    out = {}
    if BIG_JSONL:
        for r in jl(BIG_JSONL):
            out[r["run"]] = (r.get("sonuc"), r.get("gunluk") or [], r.get("olcum"))
    else:
        for d in sorted(os.listdir(BM)):
            if not os.path.isdir(os.path.join(BM, d)):
                continue
            p = os.path.join(BM, d, "sonuc.json")
            g = os.path.join(BM, d, "gunluk.jsonl")
            o = os.path.join(BM, d, "olcum.json")
            out[d] = (json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None, jl(g) if os.path.exists(g) else [],
                      json.load(open(o, encoding="utf-8")) if os.path.exists(o) else None)
    return out


BIG = {}
RUNS = load_runs()
for d, (s, grows, _olc) in sorted(RUNS.items()):
    if d[:2] in ("S1", "S2", "S3") and s is not None:
        v = [x for x in grows if "val_nat" in x]
        son_val = v[-1]["val_nat"] if v else None
        BIG[d] = dict(s, ad=d, gunluk_son_val=son_val)
print("buyuk merdiven kosu sayisi:", len(BIG))
# alt gorev (downstream) cumlesi: makale "ARC-Easy farki 3.6 puan, digerlerinde en cok 1.2 puan" diyor (olcum.json: S3_kanit - S3_taban, acc)
_tk, _tt = RUNS["S3_kanit"][2]["testler"], RUNS["S3_taban"][2]["testler"]
ALT = {k: 100 * (_tk[k]["acc"] - _tt[k]["acc"]) for k in _tk}
print("alt gorev farklari (puan, S3_kanit - S3_taban, acc):", {k: round(v, 1) for k, v in ALT.items()})
assert abs(abs(ALT["arc_easy"]) - 3.6) < 0.05 and max(abs(v) for k, v in ALT.items() if k != "arc_easy") < 1.25, "alt gorev cumlesi makaleyle uyusmuyor"
gmax = max(abs(b["val_nat"] - b["gunluk_son_val"]) for b in BIG.values() if b["gunluk_son_val"] is not None)
print("sonuc.json val_nat - gunluk son dogrulama farki (en buyuk): %.5f" % gmax)
NOT.append("sonuc.json val_nat ile gunluk.jsonl'deki son doğrulama (aynı koşuda, 0–3 adım önce) arasındaki en büyük fark %.5f nat (gürültü düzeyi; tutarlı)." % gmax)

pt = lambda ad: (BIG[ad]["flop"], BIG[ad]["val_nat"])
SIZES = ["S1", "S2", "S3"]
b_def3 = slope([BIG[s + "_taban"]["flop"] for s in SIZES], [BIG[s + "_taban"]["val_nat"] for s in SIZES])
b_def2 = slope([BIG[s + "_taban"]["flop"] for s in SIZES[:2]], [BIG[s + "_taban"]["val_nat"] for s in SIZES[:2]])
b_tun3 = slope([BIG[s + "_taban_lr2.8"]["flop"] for s in SIZES], [BIG[s + "_taban_lr2.8"]["val_nat"] for s in SIZES])
b_tun2 = slope([BIG[s + "_taban_lr2.8"]["flop"] for s in SIZES[:2]], [BIG[s + "_taban_lr2.8"]["val_nat"] for s in SIZES[:2]])
print("egimler: b_varsayilan(S1-S3)=%.4f  b_varsayilan(S1-S2)=%.4f  b_ayarli(S1-S3)=%.4f  b_ayarli(S1-S2)=%.4f" % (b_def3, b_def2, b_tun3, b_tun2))
kars("b varsayilan taban S1-S3", b_def3, 0.0595, 0.0001)
kars("b varsayilan taban S1-S2", b_def2, 0.0609, 0.0001)

# ------------------------------------------------------------------ kucuk merdiven
rows = [r for r in jl(P_SMALL) if r["arch"] == "gpt"]


def flop_fb(r):
    k = set(r["kol"].split("+"))
    return 6.0 * (r["N"] * (2 if "dongu2" in k else 1) + V * r["d"] * (2 if "mtp2" in k else 1)) * r["tokens"]


def fl(r):
    return float(r["flop"]) if r.get("flop") else flop_fb(r)


sade = [r for r in rows if r.get("flop") and not (set(r["kol"].split("+")) & {"buyuyen", "moe8", "hub4", "rholoss", "ep4", "mtp2", "dongu2"})]
fdiff = max(abs(flop_fb(r) / r["flop"] - 1) for r in sade) if sade else 0.0
print("FLOP yedek formulu vs kayitli (sade kollar, n=%d): en buyuk bagil fark %.2e" % (len(sade), fdiff))
NOT.append("FLOP yedek formülü (6·(N+V·d)·token) kayıtlı flop alanıyla sade kollarda %.1e göreli farkla aynı (n=%d); flop alanı olmayan 36 eski satır bu formülle hesaplandı." % (fdiff, len(sade)))

tab = {}
for r in rows:
    if r["kol"] == "taban" and r["tohum"] in (1, 2) and r.get("derle") is not False and r["boy"] in ("g1", "g2", "g3", "g4"):
        tab.setdefault(r["boy"], []).append((fl(r), r["val_nat"]))
TB = {b: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for b, v in tab.items()}
b_small = slope([TB[b][0] for b in ("g2", "g3", "g4")], [TB[b][1] for b in ("g2", "g3", "g4")])
print("kucuk merdiven varsayilan taban egimi g2-g4: %.5f" % b_small)
kars("b kucuk merdiven g2-g4", b_small, 0.09315, 0.0002)

t1 = [r for r in jl(P_T1) if r["etiket"] == "t1 lr2.0 wd0.1"]
TT = {r["boy"]: (fl(r), r["val_nat"]) for r in t1}
b_ts = slope([TT[b][0] for b in ("g2", "g3", "g4")], [TT[b][1] for b in ("g2", "g3", "g4")])
print("kucuk merdiven AYARLI taban egimi g2-g4: %.5f" % b_ts)

PAK = "canon+muonw+qknorm+bigram+wsd"
P2 = "canon+muonh+qknorm+bigram+wsd+buyuyen"


def sat(kol, boy, tohum, etiket):
    r = [x for x in rows if x["kol"] == kol and x["boy"] == boy and x["tohum"] == tohum and x.get("etiket") == etiket]
    return (fl(r[-1]), r[-1]["val_nat"], r[-1]) if r else None


SMALL = {}   # (ad, boy, tohum) -> dict
for ad, kol, et1, et2 in (("A", PAK, "hiz_paket", "hiz_dogrulama"), ("B", P2, "birlesim", "birlesim")):
    for tohum, et in ((1, et1), (2, et2)):
        for boy in ("g2", "g3", "g4"):
            s = sat(kol, boy, tohum, et)
            if not s:
                continue
            md = m_of(TB[boy], s[:2], b_small)
            mt = m_of(TT[boy], s[:2], b_ts)
            SMALL[(ad, boy, tohum)] = dict(loss=s[1], flop=s[0], m_def=md, m_tun=mt, N=s[2]["N"],
                                           gap_tun=(TT[boy][1] - s[1]) / TT[boy][1])
rep_def = {("A", "g2", 1): 3.411, ("A", "g3", 1): 4.898, ("A", "g3", 2): 4.882, ("A", "g4", 1): 2.207, ("A", "g4", 2): 2.185,
           ("B", "g2", 1): 3.784, ("B", "g3", 1): 5.273, ("B", "g3", 2): 5.227, ("B", "g4", 1): 2.458, ("B", "g4", 2): 2.452}
rep_tun = {("A", "g2", 1): 3.098, ("A", "g3", 1): 3.113, ("A", "g3", 2): 3.104, ("A", "g4", 1): 1.795, ("A", "g4", 2): 1.778,
           ("B", "g2", 1): 3.433, ("B", "g3", 1): 3.365, ("B", "g3", 2): 3.337, ("B", "g4", 1): 2.008, ("B", "g4", 2): 2.003}
for k, v in rep_def.items():
    kars("kucuk m varsayilan %s %s t%d" % k, SMALL[k]["m_def"], v, 0.002)
for k, v in rep_tun.items():
    kars("kucuk m ayarli %s %s t%d" % k, SMALL[k]["m_tun"], v, 0.002)

# ------------------------------------------------------------------ buyuk merdiven m tablosu
RUNS = [("A", "S1_paket", "S1"), ("A", "S1_paket_t2", "S1"), ("B", "S1_w_p2", "S1"), ("B", "S1_w_p2_t2", "S1"),
        ("A", "S2_paket", "S2"), ("B", "S2_w_p2", "S2"), ("B", "S3_kanit", "S3")]
EXTRA = [("b", "S1_b0.5", "S1"), ("b", "S1_b1.0", "S1"), ("b", "S1_b2.0", "S1"), ("h", "S1_h0.008", "S1"), ("h", "S1_h0.016", "S1"),
         ("b", "S2_b2.0", "S2")]
BT = {}
for ad, run, sz in RUNS + EXTRA:
    r = BIG[run]
    bd, bt_ = pt(sz + "_taban"), pt(sz + "_taban_lr2.8")
    BT[run] = dict(ad=ad, sz=sz, N=r["N"], loss=r["val_nat"], flop=r["flop"],
                   m_def059=m_of(bd, pt(run), b_def3), m_tun059=m_of(bt_, pt(run), b_def3),
                   m_tun_local=m_of(bt_, pt(run), b_tun3), m_tun093=m_of(bt_, pt(run), b_small),
                   m_def093=m_of(bd, pt(run), b_small),
                   c=bt_[0] / r["flop"], gap_tun=(bt_[1] - r["val_nat"]) / bt_[1], gap_def=(bd[1] - r["val_nat"]) / bd[1],
                   dnat_tun=bt_[1] - r["val_nat"])
# raporlanan degerler (RAPOR_DEVRIM_OTOMATIK.md, b=0,0595): m varsayilan / m ayarli
rep059 = {"S1_paket": (2.949, 1.963), "S1_paket_t2": (2.974, 1.980), "S1_w_p2": (3.852, 2.565), "S1_w_p2_t2": (3.791, 2.524),
          "S2_paket": (2.479, 1.952), "S2_w_p2": (3.221, 2.536), "S3_kanit": (2.765, 2.212), "S1_b0.5": (2.032, 1.353),
          "S1_b1.0": (2.958, 1.969), "S1_b2.0": (3.584, 2.386), "S1_h0.008": (3.048, 2.030), "S1_h0.016": (2.453, 1.633), "S2_b2.0": (2.866, 2.256)}
for k, (md, mt) in rep059.items():
    kars("buyuk m varsayilan b0.0595 " + k, BT[k]["m_def059"], md, 0.003)
    kars("buyuk m ayarli b0.0595 " + k, BT[k]["m_tun059"], mt, 0.003)
# F2 raporu (b=0,0932): ayarli taban m
rep093 = {"S1_paket": 1.538, "S1_w_p2": 1.991, "S1_w_p2_t2": 1.971, "S1_b2.0": 1.901}
for k, v in rep093.items():
    kars("buyuk m ayarli b0.0932 (F2 raporu) " + k, BT[k]["m_tun093"], v, 0.003)
# 7bd: 85M K4' iki egim
kars("S3_kanit m ayarli b0.0932 (7bd 1.826)", BT["S3_kanit"]["m_tun093"], 1.826, 0.003)
kars("S3_kanit m varsayilan b0.0595 (K4 2.765)", BT["S3_kanit"]["m_def059"], 2.765, 0.002)
for sz in SIZES:   # varsayilan taban m (ayarli taban, varsayilan tabana gore) rapor tablosu: 1.502 / 1.270 / 1.250
    pass
kars("S1 ayarli taban / varsayilan taban m", m_of(pt("S1_taban"), pt("S1_taban_lr2.8"), b_def3), 1.502, 0.003)
kars("S2 ayarli taban / varsayilan taban m", m_of(pt("S2_taban"), pt("S2_taban_lr2.8"), b_def3), 1.270, 0.003)
kars("S3 ayarli taban / varsayilan taban m", m_of(pt("S3_taban"), pt("S3_taban_lr2.8"), b_def3), 1.250, 0.003)

# ------------------------------------------------------------------ egim duyarliligi (paket2, ayarli taban)
BGRID = [0.045, 0.050, round(b_tun3, 4), 0.0595, 0.070, 0.080, 0.0932, 0.100]
SENS = {}
for run in ("S1_w_p2", "S1_w_p2_t2", "S2_w_p2", "S3_kanit", "S1_paket", "S2_paket"):
    sz = run[:2]
    SENS[run] = {("%.4f" % b): m_of(pt(sz + "_taban_lr2.8"), pt(run), b) for b in BGRID}
# esik 2,0'i gecmek icin gereken en buyuk egim (85M)
fl_ = pt("S3_taban_lr2.8"); rn = pt("S3_kanit")
b_esik = math.log(fl_[1] / rn[1]) / math.log(2.0 * rn[0] / fl_[0])
print("S3_kanit ayarli taban m>=2,0 esigini asmak icin egim <= %.4f olmali (ayarli taban S1-S3 yerel egim %.4f)" % (b_esik, b_tun3))

# ------------------------------------------------------------------ tohum gurultusu (S1)
def dn(a, b_): return abs(BIG[a]["val_nat"] - BIG[b_]["val_nat"])
SEED = dict(paket=dn("S1_paket", "S1_paket_t2"), paket2=dn("S1_w_p2", "S1_w_p2_t2"))
print("S1 tohum farki (nat): paket %.5f, paket2 %.5f" % (SEED["paket"], SEED["paket2"]))

# ------------------------------------------------------------------ F1A (a_k) ve T2
f1a_rep = json.load(open(P_F1A, encoding="utf-8"))
KOLMAP = {"muonh": ("canon+muonh+qknorm+bigram+wsd", "f1a"), "buyuyen": (PAK + "+buyuyen", "f1a"), "muonh+buyuyen": (P2, "f1a"),
          "overtok": ("canon+muonw+qknorm+overtok+wsd", "f1a"), "hub4": (PAK + "+hub4", "f1a"), "derin": (PAK + "+derin", "f1a"),
          "gptbert": (PAK + "+gptbert", "f1a"), "rholoss": (PAK + "+rholoss", "f1a")}
F1A = {}
for ad, (kol, et) in KOLMAP.items():
    F1A[ad] = {}
    for boy in ("g2", "g3", "g4"):
        s = sat(kol, boy, 1, et)
        p = sat(PAK, boy, 1, "hiz_paket")
        if s and p:
            a = m_of(TB[boy], s[:2], b_small) / m_of(TB[boy], p[:2], b_small)
            F1A[ad][boy] = a
            rep = f1a_rep[ad]["olcu"].get(boy, {}).get("a")
            if rep is not None:
                kars("F1A a_k %s %s" % (ad, boy), a, rep, 0.002)
T2 = {}
for k in ("taban+ep4", PAK + "+ep4"):
    for boy in ("g3", "g4"):
        s = sat(k, boy, 1, "t2")
        if s:
            T2[(k, boy)] = m_of(TB[boy], s[:2], b_small)
kars("T2 taban+ep4 g3 m", T2[("taban+ep4", "g3")], 0.901, 0.002)
kars("T2 taban+ep4 g4 m", T2[("taban+ep4", "g4")], 0.878, 0.002)
kars("T2 paket+ep4 g3 m", T2[(PAK + "+ep4", "g3")], 2.275, 0.002)
kars("T2 paket+ep4 g4 m", T2[(PAK + "+ep4", "g4")], 1.264, 0.002)
ratio_t2 = {b: T2[(PAK + "+ep4", b)] / SMALL[("A", b, 1)]["m_def"] for b in ("g3", "g4")}
# tekrarda paketin tabana ustunlugu: paket+ep4 / taban+ep4 (ayni 4 tur kosulunda)
ust_ep4 = {b: T2[(PAK + "+ep4", b)] / T2[("taban+ep4", b)] for b in ("g3", "g4")}
ust_tek = {b: SMALL[("A", b, 1)]["m_def"] for b in ("g3", "g4")}
print("T2: paket ustunlugu tek tur g3 %.2f g4 %.2f | 4 tur g3 %.2f g4 %.2f" % (ust_tek["g3"], ust_tek["g4"], ust_ep4["g3"], ust_ep4["g4"]))

# ------------------------------------------------------------------ ilk nesil adaylar (kucuk merdiven, tohum 1, varsayilan taban)
def sat_any(kol, boy, tohum=1):
    r = [x for x in rows if x["kol"] == kol and x["boy"] == boy and x["tohum"] == tohum]
    return (fl(r[-1]), r[-1]["val_nat"]) if r else None


FIRST_SPEC = [  # (etiket, kol, rapor g2, g3, g4, hukum)
    ("flat low-LR schedule (LR$\\times$0.1, init$\\times$0.3)", "duz_dusuk", (0.235, 0.239, 0.124), "killed"),
    ("sequence-length curriculum", "seq_mufredat", (1.029, 1.006, 0.856), "killed"),
    ("multi-token prediction ($t{+}2$ head)", "mtp2", (0.581, 0.701, 0.778), "killed"),
    ("data curriculum (easy to hard)", "veri_mufredat", (0.993, 0.948, 0.871), "killed"),
    ("probability-weighted loss (ZPD)", "zpd", (0.306, 0.272, 0.222), "killed"),
    ("data pruning (middle 50\\%)", "veri_budama", (1.003, 1.075, 1.001), "no gain"),
    ("plain Muon (no weight decay)", "muon", (2.545, 2.667, 0.519), "collapsed at g4"),
    ("MuonW (added post hoc)", "muonw", (1.217, 2.054, 1.402), "kept"),
    ("canon layers alone", "canon", (None, 1.857, 1.176), "kept"),
    ("bigram soft targets", "ngram", (None, 0.988, None), "stopped at g3 gate"),
    ("early growth (default LR)", "buyuyen", (None, 1.066, None), "stopped at g3 gate"),
    ("looped depth (2 passes)", "dongu2", (None, 0.605, None), "stopped at g3 gate"),
]
FIRST = {}
for lab, kol, rep, hk in FIRST_SPEC:
    FIRST[kol] = {}
    for boy, rv in zip(("g2", "g3", "g4"), rep):
        s = sat_any(kol, boy)
        if s is None:
            FIRST[kol][boy] = None
            continue
        m = m_of(TB[boy], s, b_small)
        FIRST[kol][boy] = m
        if rv is not None:
            kars("ilk nesil m %s %s" % (kol, boy), m, rv, 0.003)

# ------------------------------------------------------------------ F1B
ak = [r for r in jl(P_AK) if not r.get("oy_satiri") and not r.get("duman")]
AK = {(r["boy"], r["kol"]): r for r in ak}
F1B = {}
for boy in ("b1", "b2", "b3"):
    t = AK[(boy, "taban")]["sinav"]["OOD_ort"]
    F1B[boy] = dict(taban=t, paket=AK[(boy, "paket")]["sinav"]["OOD_ort"], dongu=AK[(boy, "dongu")]["sinav"]["OOD_ort"],
                    id_taban=AK[(boy, "taban")]["sinav"]["ID_ort"], id_paket=AK[(boy, "paket")]["sinav"]["ID_ort"],
                    tur1=AK[(boy, "dongu")]["tur1"]["OOD_ort"], tur2=AK[(boy, "dongu")]["tur2"]["OOD_ort"],
                    tur4=AK[(boy, "dongu")]["tur4"]["OOD_ort"])
for boy, v in zip(("b1", "b2", "b3"), (2.2, 6.9, 7.5)):
    kars("F1B paket-taban OOD puan " + boy, 100 * (F1B[boy]["paket"] - F1B[boy]["taban"]), v, 0.06)
for boy, v in zip(("b1", "b2", "b3"), (1.1, 8.4, 4.8)):
    kars("F1B dongu-taban OOD puan " + boy, 100 * (F1B[boy]["dongu"] - F1B[boy]["taban"]), v, 0.06)
for boy, v in zip(("b1", "b2", "b3"), (11.9, 22.6, 15.1)):
    kars("F1B dongu tur1->4 OOD puan " + boy, 100 * (F1B[boy]["tur4"] - F1B[boy]["tur1"]), v, 0.06)
oy = [r for r in jl(P_AK) if r.get("oy_satiri")]
F1B_OY = oy[-1]["oy8"]["OOD_oy8_ort"] if oy else None
kars("F1B self-consistency b3 paket", F1B_OY, 0.490, 0.001)
# not: dongu OOD_ort (tur 2) 'sinav' alani; tur4 < tur2 oldugu icin tur 1->4 artisi tur 1->2'den kucuktur
print("F1B dongu tur1/tur2/tur4 OOD:", {b: (round(F1B[b]["tur1"], 3), round(F1B[b]["tur2"], 3), round(F1B[b]["tur4"], 3)) for b in F1B})

# ------------------------------------------------------------------ F1K
bk = jl(P_BK)
F1K = {(r["kip"], r["kol"]): r for r in bk}
rep_f1k = {("tek", "taban"): (1.310e6, 0.740, 0.225), ("tek", "paket"): (2.502e6, 1.408, 0.857), ("cok5", "taban"): (1.507e6, 0.851, 0.331),
           ("cok5", "paket"): (2.648e6, 1.490, 0.999)}
for k, (b_, bp, qa) in rep_f1k.items():
    kars("F1K ogrenilen bit %s %s" % k, F1K[k]["ogrenilen_bit"] / 1e6, b_ / 1e6, 0.002)
    kars("F1K bit/param govde %s %s" % k, F1K[k]["ogrenilen_bit"] / F1K[k]["N_govde"], bp, 0.001)
    kars("F1K QA %s %s" % k, F1K[k]["qa_ort"], qa, 0.0006)

# ------------------------------------------------------------------ duvar saati (kuyruk6_durum.txt)
seg = {}
RX = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (BASLADI|SÜRÜYOR|BITTI|DURAKLATILDI) (S\d_[\w.]+)")
acik = {}
for l in open(P_QLOG, encoding="utf-8"):
    m = RX.search(l)
    if not m:
        continue
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"); ev, ad = m.group(2), m.group(3)
    if ev in ("BASLADI", "SÜRÜYOR"):
        acik[ad] = t
    elif ad in acik:
        seg[ad] = seg.get(ad, 0.0) + (t - acik.pop(ad)).total_seconds()
WALL = {}
for ad, b in BIG.items():
    if ad in seg:
        WALL[ad] = dict(duvar_sa=seg[ad] / 3600.0, sure_sn_sa=b["sure_sn"] / 3600.0)
for ad in ("S3_taban", "S3_kanit"):
    if abs(WALL[ad]["duvar_sa"] - WALL[ad]["sure_sn_sa"]) > 0.2:
        NOT.append("%s: sonuc.json 'sure_sn' = %.2f sa, ama günlük zaman damgalarından toplam duvar saati %.2f sa (duraklat-devamda sayaç yalnız son parçayı sayıyor; "
                   "egit_h.py sure_sn'i kayıtlamıyor). Süre karşılaştırması için sure_sn KULLANILMAZ." % (ad, WALL[ad]["sure_sn_sa"], WALL[ad]["duvar_sa"]))
    print("%s: sure_sn %.2f sa | duvar (log) %.2f sa" % (ad, WALL[ad]["sure_sn_sa"], WALL[ad]["duvar_sa"]))
hiz_ihl = 1_698_955_264 / (BIG["S3_taban"]["sure_sn"])
NOT.append("KANIT_ON_KAYIT.md başlığındaki 'ölçülen hız ≈627.384 token/sn ⇒ ≈0,8 saat' S3_taban'ın yalnız son parçasından (%.0f sn) türetilmiş: gerçek hız ≈%.0f token/sn "
           "(1,699 milyar token / %.1f sa). Ön kayıt dosyası değişmez (sha256); bu bir yanlış türetilmiş yardımcı sayıdır. Öngörüler (bpb bandı) FLOP'tan türediği için etkilenmez." %
           (BIG["S3_taban"]["sure_sn"], 1_698_955_264 / (WALL["S3_taban"]["duvar_sa"] * 3600), WALL["S3_taban"]["duvar_sa"]))

# ------------------------------------------------------------------ rapor ve ciktilar
tamam = sum(1 for g in GOZLEM if g[4])
print("\nKARSILASTIRMA: %d / %d tamam" % (tamam, len(GOZLEM)))
bad = [g for g in GOZLEM if not g[4]]
for g in bad:
    print("  UYUSMAZLIK: %s hesap %.4f rapor %.4f (tol %.4f)" % g[:4])

sayilar = dict(zaman=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), egimler=dict(b_varsayilan_S1S3=b_def3, b_varsayilan_S1S2=b_def2, b_ayarli_S1S3=b_tun3,
               b_ayarli_S1S2=b_tun2, b_kucuk_varsayilan_g2g4=b_small, b_kucuk_ayarli_g2g4=b_ts, b_esik_S3=b_esik),
               buyuk=BT, kucuk={"%s_%s_t%d" % k: v for k, v in SMALL.items()}, duyarlilik=SENS, tohum_gurultusu=SEED,
               F1A=F1A, T2={"%s|%s" % k: v for k, v in T2.items()}, F1B=F1B, F1B_oy8=F1B_OY,
               F1K={"%s|%s" % k: dict(bit=v["ogrenilen_bit"], bp=v["ogrenilen_bit"] / v["N_govde"], qa=v["qa_ort"]) for k, v in F1K.items()},
               duvar_saati=WALL, karsilastirma=dict(toplam=len(GOZLEM), tamam=tamam))
json.dump(sayilar, open(OUT_NUM, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

rp = ["# F5 sayı doğrulaması: ham dosyalardan yeniden hesap vs kuyruk raporları", "",
      "Üretim: `%s` (%s). Kuyruk kodu kullanılmadı; formül bağımsız yeniden yazıldı: m = (C_taban/C_kol)·(L_taban/L_kol)^(1/b)." % (SCRIPT, sayilar["zaman"]), "",
      "**Sonuç: %d karşılaştırmanın %d'i tutuyor, %d uyuşmazlık.**" % (len(GOZLEM), tamam, len(bad)), ""]
if bad:
    rp += ["## Uyuşmazlıklar", "", "| ad | hesap | rapor | tolerans |", "|---|---|---|---|"] + ["| %s | %.4f | %.4f | %.4f |" % g[:4] for g in bad] + [""]
rp += ["## Notlar ve gözlemler (rapor/sayaç tutarlılığı)", ""] + ["- " + n for n in NOT] + [
    "- Eğim kuralı raporlarda üç farklı: F2 raporu b = 0,0932 (küçük merdiven), F3 raporu b = 0,0609 (S1–S2 taban), F4/otomatik rapor b = 0,0595 (S1–S3 taban). "
    "Bu yüzden aynı koşunun m'si raporlar arasında değişiyor (ör. S1 paket ayarlı: 1,538 / 1,963). F5'te tek ana kural ve duyarlılık bandı verilir (bkz. paper).",
    "- Ayarlı tabanın kendi S1–S3 eğimi b = %.4f (varsayılan tabanınki %.4f). Ayarlı taban için daha tutarlı yerel eğim budur; K4' (≥ 2,0) bu eğimle %.2f çıkıyor." % (
        b_tun3, b_def3, BT["S3_kanit"]["m_tun_local"]),
    "- m ≥ 2,0 eşiğini 85M'de aşmak için ayarlı taban eğiminin b ≤ %.4f olması gerekiyor." % b_esik,
    "- S3 koşularının üç günlüğünde (S3_taban, S3_kanit, S3_taban_lr2.8) aynı adım 10.619'da eğitim kaybı ≈0,56–0,72 nat'a düşüyor (komşu adımlar ≈2,7–2,9): ortak veri akışındaki "
    "tek bir yarı-boş/tekrarlı parti; doğrulama kaybını etkilemez, ama günlük grafiği çiziliyorsa süzülmeli.",
    "- Küçük merdivende paket kolları derlemesiz (eager), varsayılan taban (8 Ekim) derlemeli koşuldu; derlemesiz–derlemeli taban farkı +0,0059 nat (g3, tek ölçüm) ⇒ küçük merdiven m'leri hafif muhafazakâr (≈%6)."]
open(OUT_REP, "w", encoding="utf-8", newline="\n").write("\n".join(rp) + "\n")


# ------------------------------------------------------------------ LaTeX tablolari
def f3(x): return "%.3f" % x
def pct(x): return "%.1f" % (100 * x)

T = []
T.append("%% Otomatik uretildi: hesap_ham.py (%s). Elle duzenlemeyin." % sayilar["zaman"])
# Tablo 1 (buyuk merdiven)
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Main ladder (25M--85M non-embedding parameters, 20 tokens/parameter, one run per cell unless noted). Loss is the final validation loss (nats/token, 1M held-out tokens); "
         r"$m$ is the compute multiplier against the \emph{tuned} baseline (AdamW, LR$\times$2.8). $m$ uses the baseline's own local slope $b=%.4f$; the bracket is the range for $b\in[%.4f,0.0932]$. "
         r"``gap'' is the relative loss reduction at equal tokens (slope-free).}\label{tab:main}" % (b_tun3, b_tun3))
T.append(r"\begin{tabular}{llrrrr}\toprule")
T.append(r"Size & Run & Loss & $m$ [range] & FLOP ratio $c$ & gap (\%) \\ \midrule")
for sz, nm in (("S1", "25M"), ("S2", "49M"), ("S3", "85M")):
    bs = pt(sz + "_taban"); bt = pt(sz + "_taban_lr2.8")
    T.append(r"%s & baseline, default LR & %.4f & -- & -- & -- \\" % (nm, bs[1]))
    T.append(r" & baseline, tuned LR & %.4f & 1 & 1 & 0 \\" % bt[1])
    for ad, run, sz2 in RUNS:
        if sz2 != sz:
            continue
        b = BT[run]
        lab = {"S1_paket": "Recipe A", "S1_paket_t2": "Recipe A (seed 2)", "S1_w_p2": "Recipe B", "S1_w_p2_t2": "Recipe B (seed 2)",
               "S2_paket": "Recipe A", "S2_w_p2": "Recipe B", "S3_kanit": "Recipe B"}[run]
        lo, hi = sorted([b["m_tun_local"], b["m_tun093"]])
        T.append(r" & %s & %.4f & %.2f [%.2f--%.2f] & %.3f & %.1f \\" % (lab, b["loss"], b["m_tun_local"], lo, hi, b["c"], 100 * b["gap_tun"]))
    if sz != "S3":
        T.append(r"\midrule")
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 2 (kucuk merdiven)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Small ladder (non-embedding parameters: g2 1.3M, g3 3.2M, g4 7.4M). $m$ against the default-LR baseline (slope $b=%.4f$) and against the tuned baseline (LR$\times$2.0, slope $b=%.4f$). "
         r"Seeds 1 and 2 where run.}\label{tab:small}" % (b_small, b_ts))
T.append(r"\begin{tabular}{llcrrr}\toprule")
T.append(r"Recipe & Size & Seed & $m$ (default baseline) & $m$ (tuned baseline) & gap vs tuned (\%) \\ \midrule")
for ad in ("A", "B"):
    for boy in ("g2", "g3", "g4"):
        for t in (1, 2):
            k = (ad, boy, t)
            if k in SMALL:
                T.append(r"Recipe %s & %s & %d & %.2f & %.2f & %.1f \\" % (ad, boy, t, SMALL[k]["m_def"], SMALL[k]["m_tun"], 100 * SMALL[k]["gap_tun"]))
    if ad == "A":
        T.append(r"\midrule")
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 3 (adaylar)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Ablations added to Recipe A (seed 1; $a_k=m(\text{variant})/m(\text{Recipe A})$ at size $g_k$, default-LR baseline). Preregistered rules: pass if $a_4\ge1.05$; kill if $a_4<1.0$.}\label{tab:cand}")
T.append(r"\begin{tabular}{lrrrl}\toprule Variant & $a_2$ & $a_3$ & $a_4$ & Verdict \\ \midrule")
NAMES = {"muonh": "Hyperball instead of MuonW", "buyuyen": "growth (half depth $\\to$ full at 50\\%)", "muonh+buyuyen": "Hyperball + growth (= Recipe B)",
         "overtok": "hashed 2/3-gram input embeddings", "hub4": "4 learned prefix tokens", "derin": "deeper-thinner (1.5$\\times$ layers)",
         "gptbert": "masked+causal hybrid objective", "rholoss": "learnability-based example selection"}
for ad in ("muonh", "buyuyen", "muonh+buyuyen", "overtok", "hub4", "derin", "gptbert", "rholoss"):
    a = F1A[ad]; a4 = a.get("g4")
    verdict = "killed" if a4 < 1.0 else ("passed" if a4 >= 1.05 else "no gain")
    if ad == "muonh+buyuyen":
        verdict = "passed (a$_4\\ge$a$_3$ in 2 seeds)"
    T.append(r"%s & %s & %s & %s & %s \\" % (NAMES[ad], f3(a["g2"]), f3(a["g3"]), f3(a["g4"]), verdict))
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 4 (F1B)
T.append("")
T.append(r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}")
T.append(r"\caption{Synthetic reasoning tasks (exact-match accuracy, mean of 4 tasks, out-of-distribution = longer chains than seen in training). Equal training FLOPs; one seed.}\label{tab:reason}")
T.append(r"\begin{tabular}{lrrrrrr}\toprule Size & Baseline OOD & Recipe A OOD & Loop OOD (2 passes) & Loop pass 1 & Loop pass 4 & A$-$base (pts) \\ \midrule")
for boy, nm in (("b1", "2$\\times$128"), ("b2", "3$\\times$192"), ("b3", "4$\\times$256")):
    v = F1B[boy]
    T.append(r"%s & %s & %s & %s & %s & %s & %+.1f \\" % (nm, f3(v["taban"]), f3(v["paket"]), f3(v["dongu"]), f3(v["tur1"]), f3(v["tur4"]), 100 * (v["paket"] - v["taban"])))
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 5 (F1K)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Knowledge capacity on synthetic biographies (50k people, 53.2 bits each, 100 exposures, single run). Bits per non-embedding parameter and exact-match QA accuracy on held-out question halves.}\label{tab:know}")
T.append(r"\begin{tabular}{llrrr}\toprule Phrasing & Model & Bits learned (M) & bits/param & QA acc. \\ \midrule")
for kip, kn in (("tek", "single template"), ("cok5", "5 paraphrases")):
    for kol, kn2 in (("taban", "baseline"), ("paket", "Recipe A")):
        v = F1K[(kip, kol)]
        T.append(r"%s & %s & %.2f & %.2f & %.3f \\" % (kn, kn2, v["ogrenilen_bit"] / 1e6, v["ogrenilen_bit"] / v["N_govde"], v["qa_ort"]))
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 6 (duyarlilik)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Sensitivity of $m$ (tuned baseline) to the slope $b$ used in $m=c\,(L_{\rm base}/L)^{1/b}$. The threshold $m\ge2$ of the preregistered transfer rule is met at 85M only for $b\le%.4f$.}\label{tab:sens}" % b_esik)
T.append(r"\begin{tabular}{l" + "r" * len(BGRID) + r"}\toprule $b$ & " + " & ".join("%.4f" % b for b in BGRID) + r" \\ \midrule")
for run, lab in (("S1_w_p2", "25M Recipe B"), ("S2_w_p2", "49M Recipe B"), ("S3_kanit", "85M Recipe B")):
    T.append(lab + " & " + " & ".join("%.2f" % SENS[run]["%.4f" % b] for b in BGRID) + r" \\")
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 7 (duvar saati, ek)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{Wall-clock hours per run (from timestamps of the run log; the laptop was shared with other use and runs were paused/resumed, so these are indicative only).}\label{tab:wall}")
T.append(r"\begin{tabular}{lrr}\toprule Size & tuned baseline & Recipe B \\ \midrule")
for sz, run in (("25M", "S1_w_p2"), ("49M", "S2_w_p2"), ("85M", "S3_kanit")):
    bt = {"25M": "S1_taban_lr2.8", "49M": "S2_taban_lr2.8", "85M": "S3_taban_lr2.8"}[sz]
    T.append(r"%s & %.1f & %.1f \\" % (sz, WALL[bt]["duvar_sa"], WALL[run]["duvar_sa"]))
T.append(r"\bottomrule\end{tabular}\end{table}")
# Tablo 8 (ilk nesil adaylar, ek)
T.append("")
T.append(r"\begin{table}[t]\centering\small")
T.append(r"\caption{First-generation variants on the small ladder (seed 1, $m$ against the default-LR baseline, slope $b=%.4f$; ``--'' = not run because an earlier gate failed). The recipes' components canon, MuonW (post hoc), QK-norm, bigram and WSD come from this stage.}\label{tab:first}" % b_small)
T.append(r"\begin{tabular}{lrrrl}\toprule Variant & g2 & g3 & g4 & Outcome \\ \midrule")
for lab, kol, rep, hk in FIRST_SPEC:
    v = FIRST[kol]
    T.append(r"%s & %s & %s & %s & %s \\" % (lab, *[("%.2f" % v[b]) if v[b] is not None else "--" for b in ("g2", "g3", "g4")], hk))
T.append(r"\bottomrule\end{tabular}\end{table}")
open(os.path.join(OUT, "tables.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(T) + "\n")
_parcalar = "\n".join(T).split("\\begin{table}[t]")[1:]
_adlar = ["tab_main", "tab_small", "tab_cand", "tab_reason", "tab_know", "tab_sens", "tab_wall", "tab_first"]
assert len(_parcalar) == len(_adlar), len(_parcalar)
for _ad, _p in zip(_adlar, _parcalar):
    open(os.path.join(OUT, _ad + ".tex"), "w", encoding="utf-8", newline="\n").write("%% generated by hesap_ham.py (do not edit by hand)\n\\begin{table}[t]" + _p.rstrip() + "\n")

# ------------------------------------------------------------------ sekil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=150)
xs = {"S1": BT["S1_w_p2"]["N"], "S2": BT["S2_w_p2"]["N"], "S3": BT["S3_kanit"]["N"]}
for ad, runs, mk, col in (("Recipe B", ["S1_w_p2", "S2_w_p2", "S3_kanit"], "o", "C0"), ("Recipe A", ["S1_paket", "S2_paket"], "s", "C1")):
    x = [BT[r]["N"] for r in runs]
    ml = [BT[r]["m_tun_local"] for r in runs]
    lo = [min(BT[r]["m_tun_local"], BT[r]["m_tun093"]) for r in runs]
    hi = [max(BT[r]["m_tun_local"], BT[r]["m_tun093"]) for r in runs]
    ax[0].plot(x, ml, mk + "-", color=col, label=ad)
    ax[0].fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
    ax[1].plot(x, [100 * BT[r]["gap_tun"] for r in runs], mk + "-", color=col, label=ad)
for r, mk in (("S1_w_p2_t2", "o"),):
    ax[0].plot([BT[r]["N"]], [BT[r]["m_tun_local"]], mk, mfc="none", color="C0")
    ax[1].plot([BT[r]["N"]], [100 * BT[r]["gap_tun"]], mk, mfc="none", color="C0")
ax[0].plot([BT["S1_paket_t2"]["N"]], [BT["S1_paket_t2"]["m_tun_local"]], "s", mfc="none", color="C1")
ax[1].plot([BT["S1_paket_t2"]["N"]], [100 * BT["S1_paket_t2"]["gap_tun"]], "s", mfc="none", color="C1")
for ad_, mk_, col_ in (("A", "s", "C1"), ("B", "o", "C0")):   # kucuk merdiven (tohum 1): ayri boru hatti, ayni egimsiz olcu; soluk cizgi
    xs_ = [SMALL[(ad_, b_, 1)]["N"] for b_ in ("g2", "g3", "g4")]
    ys_ = [100 * SMALL[(ad_, b_, 1)]["gap_tun"] for b_ in ("g2", "g3", "g4")]
    ax[1].plot(xs_, ys_, mk_ + ":", color=col_, alpha=0.55, ms=4)
ax[1].text(1.3e6, 6.0, "small ladder\n(different batch/val\nsettings)", fontsize=7, color="gray")
for ax_ in ax:
    ax_.set_xscale("log"); ax_.grid(True, which="both", alpha=0.3); ax_.set_xlabel("non-embedding parameters")
    ax_.minorticks_off()
ax[0].set_xticks([25e6, 49e6, 85e6]); ax[0].set_xticklabels(["25M", "49M", "85M"])
ax[1].set_xticks([1.3e6, 3.2e6, 7.4e6, 25e6, 49e6, 85e6]); ax[1].set_xticklabels(["1.3M", "3.2M", "7.4M", "25M", "49M", "85M"], fontsize=7)
ax[0].axhline(2.0, color="gray", ls=":", lw=1); ax[0].text(25.5e6, 2.03, "preregistered threshold m = 2", fontsize=7, color="gray")
ax[0].set_ylabel("compute multiplier m vs tuned baseline"); ax[0].set_title("(a) m (band: slope b = %.3f to 0.093)" % b_tun3, fontsize=9)
ax[1].set_ylabel("loss reduction at equal tokens (%)"); ax[1].set_title("(b) slope-free relative loss gap", fontsize=9); ax[1].set_ylim(0, 12)
ax[0].legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_m_size.pdf")); fig.savefig(os.path.join(OUT, "fig_m_size.png"))
print("yazildi: %s, %s, tables.tex, fig_m_size.pdf/png" % (os.path.basename(OUT_NUM), os.path.basename(OUT_REP)))
