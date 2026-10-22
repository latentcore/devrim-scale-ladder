# -*- coding: utf-8 -*-
"""F1B — akıl yürütme çekirdeği (Devrim, 12 Ekim 2026; Mert'in 10 Ekim fikrinin ölçülebilir hâli: "akıl yürütmeyi küçük
çekirdekte kanıtla"). Ön kayıt: ON_KAYIT_MERDIVEN_2026-10-08.md Ek 7 madde 5 + Ek 7a (sınav hash'leri ve kesin bütçe).

Görevler (Python üretir, cevap kesin): ARIT aritmetik zinciri (mod 100; +, −, ×), KHOP k-atlama (10 sembollük rastgele
permütasyonda k adım), LISTE liste programı izi (S sırala, R ters, U tekrarsız, T döndür), DYCK iki parantez türü (dengeli mi).
Eğitim k ≤ 4 (Dyck derinlik ≤ 4); sınav: ID (k ≤ 4, yeni örnek) ve OOD (k = 5–8). Sınav kümeleri eğitimden önce ayrı tohumla
üretilir; eğitim üreteci sınav sorularını reddeder (kesişim 0).
Kollar: taban (AdamW, GELU, kosinüs) · paket (canon+muonw+qknorm+bigram+wsd) · dongu (paket + aynı bloklar eğitimde 2 tur;
sınavda 1, 2, 4 tur). Kollar arasında eğitim hesabı eşit: dongu yarı adım. Boylar b1 2×128, b2 3×192, b3 4×256.
Güvenli duraklat: KUYRUK6_DUR (adım sonunda checkpoint, kod 3; yeniden başlatınca devam).
Kullanım: python akil_cekirdek.py --sinav (yalnız sınav kümeleri + hash) | --hepsi (bütün koşular + rapor) | --duman"""
import argparse, collections, hashlib, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BURA = os.path.dirname(os.path.abspath(__file__))
MER = os.path.join(os.path.dirname(BURA), "d1_frank", "merdiven")
sys.path.insert(0, MER)
import merdiven as M  # noqa: E402

CH = ["\x00", "^", "|", "."] + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz") + list("+-*=?>,;:()[]") \
     + list("ADEHKLRSTU")
STOI = {c: i for i, c in enumerate(CH)}
VC, PAD, BOS, SEP, EOS = len(CH), 0, 1, 2, 3
T = 64
BOYLAR = {"b1": (2, 128), "b2": (3, 192), "b3": (4, 256)}
KOLLAR = ["taban", "paket", "dongu"]
GOREVLER = ["ARIT", "KHOP", "LISTE", "DYCK"]
B = 256                      # satır / adım (64 jeton) → 16.384 jeton / adım
ADIM = int(os.environ.get("AKIL_ADIM", "9000"))  # taban ve paket; dongu yarısı (eşit hesap)
N_SINAV = 500                # görev × küme başına soru
VERI_TOHUM = 7               # eğitim verisi bütün kollarda ve boylarda aynı akış
SONUC = os.path.join(BURA, "akil_sonuc.jsonl")
DURUM = os.path.join(BURA, "akil_durum.txt")
CKPT = os.path.join(BURA, "ckpt")
DUR = os.environ.get("KUYRUK6_DUR")
CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEV = "cuda" if CUDA else "cpu"


def log(m):
    s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    try:
        print(s, flush=True)
    except Exception:  # noqa: BLE001
        pass
    with open(DURUM, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


# ------------------------------------------------------------------ görev üreteçleri

def g_arit(r, k):
    v = r.randrange(100); s = f"{v:02d}"
    for _ in range(k):
        op = r.choice("+-*")
        if op == "*":
            a = r.choice((2, 3)); v = (v * a) % 100
        elif op == "+":
            a = r.randrange(1, 10); v = (v + a) % 100
        else:
            a = r.randrange(1, 10); v = (v - a) % 100
        s += f"{op}{a}"
    return "A" + s + "=", f"{v:02d}"


def g_khop(r, k):
    sy = r.sample("abcdefghijklmnopqrstuvwxyz", 10)
    pe = sy[:]; r.shuffle(pe)
    nx = dict(zip(sy, pe))
    ci = [f"{a}>{nx[a]}" for a in sy]; r.shuffle(ci)
    st = r.choice(sy); c = st
    for _ in range(k):
        c = nx[c]
    return "K" + ",".join(ci) + ";" + st + str(k) + "?", c


def g_liste(r, k):
    L = [r.randrange(10) for _ in range(6)]
    ops = "".join(r.choice("SRUT") for _ in range(k))
    x = L[:]
    for o in ops:
        if o == "S":
            x = sorted(x)
        elif o == "R":
            x = x[::-1]
        elif o == "U":
            x = list(dict.fromkeys(x))
        else:
            x = x[1:] + x[:1]
    return "L" + "".join(map(str, L)) + ":" + ops, "".join(map(str, x))


def _dengeli(s):
    st = []
    for c in s:
        if c in "([":
            st.append(c)
        else:
            if not st or {"(": ")", "[": "]"}[st.pop()] != c:
                return False
    return not st


def _derinlik(s):
    d = m = 0
    for c in s:
        d += 1 if c in "([" else -1; m = max(m, d)
    return m


def _dyck(r, n, tavan):
    out, st, acik = [], [], n
    for _ in range(2 * n):
        if acik > 0 and len(st) < tavan and (not st or r.random() < 0.5):
            t = r.choice("(["); out.append(t); st.append(t); acik -= 1
        elif st:
            t = st.pop(); out.append(")" if t == "(" else "]")
        else:  # tavan yüzünden açılamadı ama yığın boş: aç
            t = r.choice("(["); out.append(t); st.append(t); acik -= 1
    while st:
        t = st.pop(); out.append(")" if t == "(" else "]")
    return "".join(out)


def g_dyck(r, d):
    s = None
    for _ in range(200):
        n = r.randint(d, min(2 * d, 14))
        c = _dyck(r, n, d)
        if _derinlik(c) == d and len(c) <= 28:
            s = c; break
    if s is None:  # güvence: omurga + rastgele eşleşme
        s = "".join(r.choice("([") for _ in range(d)); s = s + "".join(")" if c == "(" else "]" for c in reversed(s))
    if r.random() < 0.5:
        return "D" + s, "E"
    for _ in range(100):
        t = list(s); i = r.randrange(len(t)); tur = r.randrange(3)
        if tur == 0:
            t[i] = {"(": "[", "[": "(", ")": "]", "]": ")"}[t[i]]
        elif tur == 1 and len(t) > 2:
            del t[i]
        elif i + 1 < len(t) and t[i] != t[i + 1]:
            t[i], t[i + 1] = t[i + 1], t[i]
        t = "".join(t)
        if not _dengeli(t):
            return "D" + t, "H"
    return "D" + s[:-1], "H"


URET = {"ARIT": g_arit, "KHOP": g_khop, "LISTE": g_liste, "DYCK": g_dyck}


def ornek(r, gorev, k):
    return URET[gorev](r, k)


# ------------------------------------------------------------------ sınav kümeleri (eğitimden ÖNCE; Ek 7a)

def sinav_kumeleri(yaz=True):
    """{(görev, 'ID'|'OOD'): [(soru, cevap, k), ...]} — ID tohum 1001+, OOD tohum 2001+; ID k 1–4, OOD k 5–8 (k rastgele;
    tekrar eden soru atlanır — küçük k'da benzersiz soru az olduğu için o k'lar biraz daha az temsil edilir)."""
    S = {}
    for gi, g in enumerate(GOREVLER):
        for kume, t0, ks in (("ID", 1001, (1, 2, 3, 4)), ("OOD", 2001, (5, 6, 7, 8))):
            r = random.Random(t0 + gi); gor = set(); out = []
            deneme = 0
            while len(out) < N_SINAV:
                deneme += 1
                assert deneme < 200 * N_SINAV, f"{g} {kume}: benzersiz soru üretilemedi"
                k = r.choice(ks)
                q, a = ornek(r, g, k)
                if q in gor:
                    continue
                gor.add(q); out.append((q, a, k))
            S[(g, kume)] = out
    if yaz:
        d = os.path.join(BURA, "sinav"); os.makedirs(d, exist_ok=True)
        hs = {}
        for (g, kume), v in S.items():
            p = os.path.join(d, f"{g}_{kume}.tsv")
            txt = "".join(f"{q}\t{a}\t{k}\n" for q, a, k in v)
            open(p, "w", encoding="utf-8", newline="\n").write(txt)
            hs[f"{g}_{kume}"] = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        open(os.path.join(d, "SHA256.json"), "w", encoding="utf-8").write(json.dumps(hs, indent=1))
    return S


def kodla(q, a=None):
    x = [BOS] + [STOI[c] for c in q] + [SEP]
    if a is not None:
        x += [STOI[c] for c in a] + [EOS]
    return x


def egitim_verisi(n_satir, yasak, tohum=VERI_TOHUM):
    """(x, y, maske) numpy: satır başına bir örnek, sağdan dolgu; kayıp yalnız cevap jetonlarında (ve EOS'ta)."""
    r = random.Random(tohum)
    X = np.zeros((n_satir, T + 1), dtype=np.uint8); Mk = np.zeros((n_satir, T), dtype=bool)
    red = 0; i = 0
    while i < n_satir:
        g = GOREVLER[r.randrange(4)]; k = r.randint(1, 4)
        q, a = ornek(r, g, k)
        if q in yasak:
            red += 1; continue
        z = kodla(q, a)
        X[i, :len(z)] = z
        bas = len(q) + 1  # SEP konumu: y[SEP] = ilk cevap jetonu
        Mk[i, bas:bas + len(a) + 1] = True
        i += 1
    return X, Mk, red


# ------------------------------------------------------------------ model

class AkilModel(nn.Module):
    def __init__(self, L, d, kol):
        super().__init__()
        pk = kol in ("paket", "dongu")
        self.emb = nn.Embedding(VC, d)
        h = max(1, d // 64)
        self.bloklar = nn.ModuleList([M.Blok(d, h, canon=pk, qknorm=pk) for _ in range(L)])
        self.nf = M.RMSNorm(d); self.bas = nn.Linear(d, VC, bias=False)
        self.bg = nn.Embedding(VC * VC, d) if pk else None
        self.tekrar = 2 if kol == "dongu" else 1
        hd = d // h
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        fr = torch.outer(torch.arange(T + 16).float(), inv)
        self.register_buffer("cos", fr.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", fr.sin()[None, :, None, :], persistent=False)

    def forward(self, x, tekrar=None):
        Tn = x.size(1); cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        h = self.emb(x)
        if self.bg is not None:
            h = h + self.bg(F.pad(x[:, :-1], (1, 0)) * VC + x)
        for _ in range(tekrar or self.tekrar):
            for b in self.bloklar:
                h = b(h, cos, sin)
        return self.bas(self.nf(h))


def govde(m):
    return sum(p.numel() for n, p in m.named_parameters() if n.startswith("bloklar"))


@torch.no_grad()
def cevapla(model, sorular, tekrar=None, sicaklik=0.0, n_ornek=1, tohum=0):
    """Açgözlü (ya da sıcaklıklı) çözme; aynı uzunluktaki sorular birlikte. Döner: her soru için cevap listesi (n_ornek)."""
    model.eval()
    g = torch.Generator(device=DEV); g.manual_seed(tohum)
    grup = collections.defaultdict(list)
    for i, q in enumerate(sorular):
        grup[len(q)].append(i)
    out = [[None] * n_ornek for _ in sorular]
    for L_, idx in grup.items():
        for j in range(n_ornek):
            for p0 in range(0, len(idx), 512):
                ii = idx[p0:p0 + 512]
                x = torch.tensor([kodla(sorular[i]) for i in ii], device=DEV)
                bitti = torch.zeros(len(ii), dtype=torch.bool, device=DEV)
                uret = []
                for _ in range(10):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
                        lo = model(x, tekrar=tekrar)[:, -1].float()
                    if sicaklik > 0:
                        nx = torch.multinomial(F.softmax(lo / sicaklik, -1), 1, generator=g).squeeze(1)
                    else:
                        nx = lo.argmax(-1)
                    nx = torch.where(bitti, torch.full_like(nx, EOS), nx)
                    uret.append(nx); bitti |= nx == EOS
                    x = torch.cat([x, nx[:, None]], 1)
                    if bool(bitti.all()):
                        break
                U = torch.stack(uret, 1).tolist()
                for r_, i in enumerate(ii):
                    s = ""
                    for t in U[r_]:
                        if t == EOS:
                            break
                        s += CH[t] if 0 <= t < VC else "?"
                    out[i][j] = s
    model.train()
    return out


def degerlendir(model, S, tekrar=None):
    acc = {}
    for (g, kume), v in S.items():
        c = cevapla(model, [q for q, _, _ in v], tekrar=tekrar)
        acc[f"{g}_{kume}"] = round(sum(c[i][0] == a for i, (_, a, _) in enumerate(v)) / len(v), 4)
    acc["ID_ort"] = round(sum(acc[f"{g}_ID"] for g in GOREVLER) / 4, 4)
    acc["OOD_ort"] = round(sum(acc[f"{g}_OOD"] for g in GOREVLER) / 4, 4)
    return acc


def oy(model, S, n=8, sicaklik=0.7, tekrar=None):
    acc = {}
    for (g, kume), v in S.items():
        if kume != "OOD":
            continue
        c = cevapla(model, [q for q, _, _ in v], tekrar=tekrar, sicaklik=sicaklik, n_ornek=n, tohum=11)
        acc[f"{g}_OOD_oy8"] = round(sum(collections.Counter(c[i]).most_common(1)[0][0] == a for i, (_, a, _) in enumerate(v)) / len(v), 4)
    acc["OOD_oy8_ort"] = round(sum(acc[f"{g}_OOD_oy8"] for g in GOREVLER) / 4, 4)
    return acc


# ------------------------------------------------------------------ eğitim

def egit(boy, kol, S, X, Mk, tohum=1):
    L, d = BOYLAR[boy]
    torch.manual_seed(tohum)
    model = AkilModel(L, d, kol).to(DEV)
    M.init_(model, "gpt", 1.0, L)
    adim = ADIM // 2 if kol == "dongu" else ADIM
    lr_tepe = 1.5e-3 * 256 / d
    isinma = max(20, int(0.02 * adim))
    optlar = M.opt_kur(model, "muonw" if kol != "taban" else "taban", lr_tepe)
    wsd = kol != "taban"

    def lr_at(s):
        if s < isinma:
            return lr_tepe * (s + 1) / isinma
        if wsd:
            sg = max(1, int(0.45 * adim))
            return lr_tepe if s < adim - sg else lr_tepe * (1 - 0.9 * (s - (adim - sg)) / sg)
        p = (s - isinma) / max(1, adim - isinma)
        return lr_tepe * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    os.makedirs(CKPT, exist_ok=True)
    ck = os.path.join(CKPT, f"{boy}_{kol}_t{tohum}_{ADIM}.pt")
    s0, gecen = 0, 0.0
    if os.path.exists(ck):
        k = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(k["model"])
        for (o, _), st in zip(optlar, k["optlar"]):
            o.load_state_dict(st)
        s0, gecen = k["s"], k["gecen"]
        log(f"{boy} {kol}: DEVAM adım {s0}/{adim}")
    t0 = time.time(); son = time.time(); iz = []
    for s in range(s0, adim):
        for o, b0 in optlar:
            for g in o.param_groups:
                g["lr"] = lr_at(s) if b0 == lr_tepe else b0 * lr_at(s) / lr_tepe
        a = torch.from_numpy(X[s * B:(s + 1) * B].astype(np.int64)).to(DEV)
        m = torch.from_numpy(Mk[s * B:(s + 1) * B]).to(DEV)
        x, y = a[:, :-1], a[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
            lo = model(x)
        loss = F.cross_entropy(lo.float()[m], y[m])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, _ in optlar:
            o.step(); o.zero_grad(set_to_none=True)
        if s % max(1, adim // 10) == 0 or s == adim - 1:
            iz.append((s, round(loss.item(), 4)))
            print(f"[{boy}/{kol}] adım {s}/{adim} kayıp {loss.item():.4f}", flush=True)
        if s < adim - 1 and DUR and (os.path.exists(DUR) or time.time() - son > 600):
            torch.save(dict(model=model.state_dict(), optlar=[o.state_dict() for o, _ in optlar], s=s + 1,
                            gecen=gecen + time.time() - t0), ck + ".tmp")
            os.replace(ck + ".tmp", ck); son = time.time()
            if os.path.exists(DUR):
                log(f"{boy} {kol}: DURAKLATILDI adım {s + 1}/{adim}")
                sys.exit(3)
    sure = gecen + time.time() - t0
    satir = dict(boy=boy, kol=kol, tohum=tohum, L=L, d=d, N_govde=govde(model), adim=adim, jeton=adim * B * T,
                 hesap=6.0 * (govde(model) * (2 if kol == "dongu" else 1) + VC * d) * adim * B * T, egitim_sn=round(sure, 1),
                 kayip_iz=iz, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
    if kol == "dongu":
        for tur in (1, 2, 4):
            satir[f"tur{tur}"] = degerlendir(model, S, tekrar=tur)
        satir["sinav"] = satir["tur2"]
    else:
        satir["sinav"] = degerlendir(model, S)
    return model, satir


# ------------------------------------------------------------------ rapor ve kapı (K1B)

def rapor(R):
    def bul(boy, kol):
        r = [x for x in R if x["boy"] == boy and x["kol"] == kol]
        return r[-1] if r else None
    sat = ["## F1B — akıl yürütme çekirdeği (Ek 7 madde 5, Ek 7a)", "",
           "| Boy | Kol | ID ort | OOD ort (uzun k) | ARIT OOD | KHOP OOD | LISTE OOD | DYCK OOD | Not |", "|---|---|---|---|---|---|---|---|---|"]
    for boy in BOYLAR:
        for kol in KOLLAR:
            x = bul(boy, kol)
            if not x:
                continue
            s = x["sinav"]
            not_ = ""
            if kol == "dongu":
                not_ = "tur 1/2/4 OOD: " + " / ".join(f"{x[f'tur{t}']['OOD_ort']:.3f}" for t in (1, 2, 4))
            sat.append(f"| {boy} | {kol} | {s['ID_ort']:.3f} | {s['OOD_ort']:.3f} | {s['ARIT_OOD']:.3f} | {s['KHOP_OOD']:.3f} | "
                       f"{s['LISTE_OOD']:.3f} | {s['DYCK_OOD']:.3f} | {not_} |")
    hk = []
    for kol in ("paket", "dongu"):
        f = []
        for boy in BOYLAR:
            a, t = bul(boy, kol), bul(boy, "taban")
            if a and t:
                f.append((boy, 100 * (a["sinav"]["OOD_ort"] - t["sinav"]["OOD_ort"])))
        if len(f) >= 2:
            gecer = f[-1][1] >= 5 and f[-1][1] >= f[0][1]
            olur = f[-1][1] <= 0
            hk.append(f"- {kol} − taban (OOD ort, puan): " + ", ".join(f"{b} {v:+.1f}" for b, v in f)
                      + f" ⇒ **{'GEÇTİ' if gecer else ('ÖLDÜ' if olur else 'geçmedi (kazanç < 5 puan ya da büyük boyda küçüldü)')}**")
    for boy in BOYLAR:
        x = bul(boy, "dongu")
        if x:
            art = 100 * (x["tur4"]["OOD_ort"] - x["tur1"]["OOD_ort"])
            hk.append(f"- {boy} dongu test anı turu 1 → 4: OOD {art:+.1f} puan ⇒ {'TEST ANI İŞARETİ' if art >= 5 else 'işaret yok'}")
    sc = [x for x in R if x.get("oy8")]
    for x in sc:
        hk.append(f"- self-consistency (8 örnek, çoğunluk) {x['boy']} {x['kol']}: OOD {x['sinav']['OOD_ort']:.3f} → {x['oy8']['OOD_oy8_ort']:.3f}")
    sat += ["", *hk]
    open(os.path.join(MER, "kuyruk6_rapor_F1B.md"), "w", encoding="utf-8", newline="\n").write("\n".join(sat) + "\n")
    return hk


def haiku_istemi(S):
    """İsteğe bağlı (Mert'in işi): OOD kümesinden 40 soruluk istem + ayrı cevap anahtarı."""
    r = random.Random(4545); sec = []
    for g in GOREVLER:
        sec += [(g, *x) for x in r.sample(S[(g, "OOD")], 10)]
    acik = ("Aşağıda 40 kısa bulmaca var. Araç kullanma, kod çalıştırma. Her satıra yalnız numara ve cevabı yaz (ör. '7: 42').\n"
            "Biçimler:\n- A: iki basamaklı sayı, sonra soldan sağa işlemler (+, −, ×), her adımdan sonra mod 100; sonucu iki basamakla yaz "
            "(ör. A47+3*2-5= → 95).\n- K: 'a>f' a'nın sonraki sembolü f demek; ';' sonrası başlangıç sembolü ve atlama sayısı; o kadar "
            "atlayınca varılan sembolü yaz.\n- L: altı rakamlık liste, ':' sonrası işlemler soldan sağa: S küçükten büyüğe sırala, R ters "
            "çevir, U tekrarları sil (ilk görüleni tut), T ilk elemanı sona taşı; sonuç listeyi bitişik rakamlarla yaz.\n- D: parantez "
            "dizisi; dengeliyse E, değilse H yaz.\n\n")
    soru = "".join(f"{i + 1}: {q}\n" for i, (g, q, a, k) in enumerate(sec))
    anahtar = "".join(f"{i + 1}: {a}  ({g}, k={k})\n" for i, (g, q, a, k) in enumerate(sec))
    open(os.path.join(BURA, "HAIKU45_AKIL_TESTI_PROMPT.txt"), "w", encoding="utf-8", newline="\n").write(acik + soru)
    open(os.path.join(BURA, "HAIKU45_AKIL_TESTI_CEVAP_ANAHTARI.txt"), "w", encoding="utf-8", newline="\n").write(anahtar)


def oku():
    return [json.loads(l) for l in open(SONUC, encoding="utf-8")] if os.path.exists(SONUC) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinav", action="store_true"); ap.add_argument("--hepsi", action="store_true"); ap.add_argument("--duman", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    S = sinav_kumeleri(yaz=True)
    if a.sinav:
        print(open(os.path.join(BURA, "sinav", "SHA256.json"), encoding="utf-8").read()); return
    yasak = {q for v in S.values() for q, _, _ in v}
    global ADIM
    if a.duman:
        ADIM = 20
    n = ADIM * B
    t0 = time.time()
    X, Mk, red = egitim_verisi(n, yasak)
    log(f"eğitim verisi: {n:,} örnek, sınavla çakışıp reddedilen {red}, {time.time() - t0:.0f} sn")
    kesisim = len({bytes(r_.tobytes()) for r_ in X[:0]})  # bilgi amaçlı; asıl güvence üreteçteki ret
    R = oku()
    for boy in BOYLAR:
        for kol in KOLLAR:
            if a.duman and boy != "b1":
                continue
            if any(x["boy"] == boy and x["kol"] == kol and x.get("adim_temel") == ADIM for x in R):
                log(f"zaten var: {boy} {kol}"); continue
            log(f"BASLADI {boy} {kol}")
            model, satir = egit(boy, kol, S, X, Mk)
            satir["adim_temel"] = ADIM; satir["reddedilen"] = red; satir["duman"] = bool(a.duman)
            log(f"BITTI {boy} {kol}: ID {satir['sinav']['ID_ort']:.3f} OOD {satir['sinav']['OOD_ort']:.3f} ({satir['egitim_sn'] / 60:.1f} dk)")
            if not a.duman:
                with open(SONUC, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(satir, ensure_ascii=False) + "\n")
                ck = os.path.join(CKPT, f"{boy}_{kol}_t1_{ADIM}.pt")
                if os.path.exists(ck):
                    os.remove(ck)
                R.append(satir)
                if boy == "b3" and kol == "paket":  # self-consistency (T4) en büyük boyda paket kolunda
                    satir2 = dict(satir); satir2["oy8"] = oy(model, S); satir2["kol"] = "paket"
                    with open(SONUC, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dict(satir2, oy_satiri=True), ensure_ascii=False) + "\n")
                    R.append(satir2)
            del model
            if CUDA:
                torch.cuda.empty_cache()
    if a.duman:
        log("DUMAN BITTI"); return
    hk = rapor([x for x in R if not x.get("duman")])
    haiku_istemi(S)
    log("F1B BITTI: " + " | ".join(hk))


if __name__ == "__main__":
    main()
