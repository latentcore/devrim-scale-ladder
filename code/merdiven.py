# -*- coding: utf-8 -*-
"""Merdiven: Devrim Aşama 4 aleti. Aynı veri ve tarifle birkaç boyutta küçük model eğitir, her boyutun
doğrulama kaybını yazar; egim.py bu noktalardan ölçekleme üssünü (α) ve kollar arası farkı (Δα) hesaplar.

Kollar (--variant): taban | duz_dusuk (Frank kaldıracı: sabit LR x0.1 + init x0.3) | seq_mufredat (kısa bağlamdan
uzuna: 64->128->256->512) | mtp2 (t+2 tahmini için ek baş, kayıp 1 + 0.5) | veri_mufredat (kolaydan zora belge sırası)
| zpd (p~0,5 tokenlara ağırlık) | veri_budama (2 kat havuzdan orta %50) | canon (pencere-4 nedensel evrişim)
| muon (gövde 2B ağırlıkları Muon) | dongu2 (aynı bloklar iki kez) | ngram (ilk %25'te hedefe bigram karışır)
| buyuyen (yarım derinlik, yarıda blok kopyalayarak tam derinlik). Tanımlar: ON_KAYIT_MERDIVEN_2026-10-08.md (Ek 2).
Speedrun tarif bayrakları (Ek 4; modded-nanogpt): qknorm | relu2 | sifir (sıfır başlangıçlı çıkış projeksiyonları) | softcap
| bigram (hash gömmesi) | wsd (sabit + son %45 iniş). Kollar "+" ile birleşir (ör. "muon+qknorm+relu2").
kuyruk6 kolları (12 Ekim 2026, ön kayıt Ek 7): muonh (Hyperball: gövde 2B ağırlık ve güncelleme Frobenius normu sabit, ağırlık
azaltma yok; η = MERDIVEN_H_TEPE) | overtok (2-gram + 3-gram hash'li giriş gömmeleri) | hub4 (dizinin başına 4 öğrenilmiş jeton)
| derin (aynı gövde parametresi, ≈1,5 kat katman) | moe8 (8 uzman, en iyi 2, uzman genişliği 2d, yük dengeleme 0,01)
| ve (her blokta değer gömmesi) | x0 (blok girişi λa·x + λb·x0) | gptbert (mikro partilerin ¼'ü maskeli, çift yönlü)
| rholoss (2 kat aday dizi, eğitim − referans kaybı en yüksek %50; referans g1 taban, train.bin'in son 40M jetonu)
| ep4 (benzersiz verinin ¼'ü × 4 tur; kontrol). Varsayılan yol eski kodla CPU'da bit bit aynıdır (sina_k6.py).
Mimari (--arch): gpt | lstm (D10 alet kalibrasyonu için). Ortam: MERDIVEN_LSTM=v2 (LSTM giriş/çıkış RMSNorm;
v1 = 8 Ekim gecesi hâli, unigram düzlüğünde kalıyordu), MERDIVEN_ADIM_MAX (yalnız duman sınaması), MERDIVEN_MIKRO, MERDIVEN_WD (AdamW ağırlık azaltması, varsayılan 0,1; kuyruk6 T1),
MERDIVEN_H_TEPE (muonh η), KUYRUK6_DUR (bu dosya varsa adım sonunda checkpoint alıp kod 3 ile çıkar), MERDIVEN_CKPT (checkpoint
klasörü; verilirse 10 dk'da bir kayıt ve yarım koşudan otomatik devam).
Veri: token_onbellek.py çıktısı (~/.cache/autoresearch/merdiven), sıra sabit; tohum yalnız başlatmayı değiştirir.
Kullanım: python merdiven.py --arch gpt --sizes g1,g2,g3,g4 --variant taban --seed 1 --out sonuclar.jsonl"""
import argparse, copy, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CC", "C:/Users/np086/AppData/Roaming/Python/Python313/site-packages/triton/runtime/tcc/tcc.exe")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
MD = os.path.join(CACHE, "merdiven")
V = 8192
HIZLI_CANON = os.environ.get("MERDIVEN_HIZLI_CANON", "0") == "1"  # hız (13 Ekim): canon kaydırmalı toplamla; varsayılan kapalı
GPT_BOY = {"g0": (2, 96), "g1": (2, 128), "g2": (3, 192), "g3": (4, 256), "g4": (6, 320), "g5": (8, 384)}
GPT_DERIN = {"g0": (3, 78), "g1": (3, 104), "g2": (5, 148), "g3": (6, 210), "g4": (9, 256)}  # derin kolu: aynı gövde, ≈1,5 kat katman
LSTM_BOY = {"l0": (2, 120), "l1": (2, 160), "l2": (2, 288), "l3": (2, 448), "l4": (2, 640), "l5": (2, 896)}
TOK_ADIM = 32768      # adım başına token (tüm boyutlar ve kollar aynı)
MIKRO = int(os.environ.get("MERDIVEN_MIKRO", "8192"))  # mikro parti token (bellek için; sonucu değiştirmez)
T_SON = 512
MUON_LR = float(os.environ.get("MERDIVEN_MUON_LR", "0.02"))  # muon kolunun 2B ağırlık LR'si (Ek 3 taraması)
AW_WD = float(os.environ.get("MERDIVEN_WD", "0.1"))  # AdamW ağırlık azaltması (kuyruk6 F0/T1 taban ayarı; varsayılan eski değer)
H_TEPE = float(os.environ.get("MERDIVEN_H_TEPE", "0.008"))  # muonh (Hyperball) göreli adım boyu η
# torch.compile: 8 Ekim 16:23'ten beri Windows Akıllı Uygulama Denetimi torchinductor'ın Temp'te ürettiği DLL'leri engelliyor
# (CodeIntegrity olay 3077/3118) → varsayılan derlemesiz (eager). Matematik aynı, yalnız daha yavaş. MERDIVEN_DERLE=1 eski hâl.
DERLE = os.environ.get("MERDIVEN_DERLE", "0") == "1"
BIGRAM_M = 131071  # bigram hash gömmesinin satır sayısı (asal; modded-nanogpt rekor 62'nin küçük hâli)
TRIGRAM_M = 262139  # overtok: 3-gram hash gömmesinin satır sayısı
HUB_K = 4          # hub4: dizinin başına eklenen öğrenilmiş jeton sayısı
MOE_E, MOE_K, MOE_AUX = 8, 2, 0.01  # moe8: uzman sayısı, jeton başına uzman, yük dengeleme katsayısı
GB_ORAN, GB_MASKE = 4, 0.15  # gptbert: her 4. mikro parti maskeli, maske oranı
RHO_HOLDOUT = 40_000_000  # rholoss: referans modelin eğitildiği, ana eğitimin dokunmadığı son jetonlar
SOFTCAP = 15.0     # logit yumuşak tavanı (modded-nanogpt rekor 18: 30 → 15)
LSTM_KIP = os.environ.get("MERDIVEN_LSTM", "v1")  # v1: 8 Ekim gecesi hâli; v2: giriş/çıkış RMSNorm (düzlük düzeltmesi)
DUR = os.environ.get("KUYRUK6_DUR")  # güvenli duraklat dosyası (kuyruk6)
CKPT_DIR = os.environ.get("MERDIVEN_CKPT")  # checkpoint klasörü (verilmezse checkpoint yok: eski davranış)
CKPT_DK = 10.0
CUDA_VAR = torch.cuda.is_available() and torch.cuda.device_count() > 0  # Windows: gizli GPU'da is_available True dönebiliyor
SINA_DUR_ADIM = None  # yalnız sina_k6.py: bu adımın sonunda DUR dosyası yaratılır (duraklat/devam sınaması)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.w)


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class MoE(nn.Module):
    """moe8: jeton başına en iyi k uzman (olasılıklar yeniden normalize), uzman genişliği 2d; Switch tarzı yük dengeleme kaybı."""

    def __init__(self, d, E=MOE_E, k=MOE_K, relu2=False):
        super().__init__()
        self.E, self.k, self.relu2 = E, k, relu2
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, d, 2 * d)); self.w2 = nn.Parameter(torch.empty(E, 2 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        p = self.router(xf).float().softmax(-1)
        topv, topi = p.topk(self.k, -1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e in range(self.E):
            m = topi == e
            satir = m.any(-1).nonzero(as_tuple=True)[0]
            if satir.numel() == 0:
                continue
            w = (topv * m).sum(-1)[satir]
            h = xf[satir] @ self.w1[e].to(xf.dtype)
            h = F.relu(h).square() if self.relu2 else F.gelu(h)
            y = h @ self.w2[e].to(h.dtype)
            out.index_add_(0, satir, (y.float() * w.unsqueeze(-1)).to(out.dtype))
        f = torch.zeros(self.E, device=x.device).index_add_(0, topi.reshape(-1), torch.ones(topi.numel(), device=x.device))
        aux = self.E * ((f / topi.numel()) * p.mean(0)).sum()
        return out.view(B, T, D), aux


class Blok(nn.Module):
    def __init__(self, d, h, canon=False, qknorm=False, relu2=False, ve=0, x0=False, moe=0):
        super().__init__()
        self.h = h; self.n1 = RMSNorm(d); self.n2 = RMSNorm(d)
        self.qknorm, self.relu2 = qknorm, relu2  # speedrun bayrakları (Ek 4); parametresiz
        # Canon (Allen-Zhu, Physics of LMs 4.1): artık akışa kısa nedensel derinlik-evrişimi (pencere 4), A ve C noktaları
        self.cA = nn.Conv1d(d, d, 4, groups=d, bias=False) if canon else None
        self.cC = nn.Conv1d(d, d, 4, groups=d, bias=False) if canon else None
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        if moe:
            self.moe = MoE(d, E=moe, relu2=relu2)
        else:
            self.moe = None
            self.f1 = nn.Linear(d, 4 * d, bias=False); self.f2 = nn.Linear(4 * d, d, bias=False)
        self.ve = nn.Embedding(ve, d) if ve else None  # değer gömmesi (speedrun "value embeddings"), sıfırdan başlar
        self.lam = nn.Parameter(torch.tensor([1.0, 0.0])) if x0 else None  # x0 karışımı (speedrun "x0 lambdas")
        self.aux = None

    @staticmethod
    def _canon(c, x):
        if HIZLI_CANON:
            return Blok._canon_hizli(c, x)
        return x + c(F.pad(x.transpose(1, 2), (3, 0))).transpose(1, 2).contiguous()

    @staticmethod
    def _canon_hizli(c, x):
        """_canon ile aynı matematik: y[t] = w3·x[t] + w2·x[t-1] + w1·x[t-2] + w0·x[t-3] (kanal başına). Autocast açıkken evrişim
        gibi bf16 girdi, fp32 toplam, bf16 çıktı. Evrişim ve transpoze yok; derleyici tek çekirdekte birleştirir."""
        ac = torch.is_autocast_enabled() and x.is_cuda
        w = c.weight.view(-1, 4)
        xb, wb = (x.to(torch.bfloat16).float(), w.to(torch.bfloat16).float()) if ac else (x, w)
        T = x.size(1)
        xp = F.pad(xb, (0, 0, 3, 0))
        y = xp[:, 3:3 + T] * wb[:, 3] + xp[:, 2:2 + T] * wb[:, 2] + xp[:, 1:1 + T] * wb[:, 1] + xp[:, :T] * wb[:, 0]
        return x + (y.to(torch.bfloat16) if ac else y)

    def forward(self, x, cos, sin, x0=None, idx=None, causal=True, hub=0):
        B, T, D = x.shape
        if self.lam is not None:
            x = self.lam[0] * x + self.lam[1] * x0
        if self.cA is not None:
            x = self._canon(self.cA, x)
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        if self.ve is not None:
            e = self.ve(idx)
            if hub:
                e = F.pad(e, (0, 0, hub, 0))
            v = v + e.view(B, T, self.h, D // self.h).to(v.dtype)
        if self.qknorm:  # QK-norm (modded-nanogpt rekor 5): baş boyutunda ağırlıksız RMSNorm
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = rope(q, cos, sin).type_as(v), rope(k, cos, sin).type_as(v)  # SDPA için tek dtype (derlemede şart)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        if self.cC is not None:
            x = self._canon(self.cC, x)
        if self.moe is not None:
            a, self.aux = self.moe(self.n2(x))
            return x + a
        a = self.f1(self.n2(x))
        a = F.relu(a).square() if self.relu2 else F.gelu(a)  # ReLU² (rekor 5)
        return x + self.f2(a)


class GPT(nn.Module):
    def __init__(self, L, d, mtp=False, canon=False, tekrar=1, qknorm=False, relu2=False, softcap=0.0, bigram=0,
                 trigram=0, hub=0, ve=False, x0=False, moe=0, maske=False):
        super().__init__()
        self.d, self.h, self.tekrar = d, max(1, d // 64), tekrar
        Vg = V + 1 if maske else V  # gptbert: ayrı maske jetonu (V)
        self.emb = nn.Embedding(Vg, d)
        self.bloklar = nn.ModuleList([Blok(d, self.h, canon, qknorm, relu2, ve=(Vg if ve else 0), x0=x0, moe=moe) for _ in range(L)])
        self.nf = RMSNorm(d); self.bas = nn.Linear(d, V, bias=False)
        self.bas2 = nn.Linear(d, V, bias=False) if mtp else None
        self.softcap = softcap
        self.bg = nn.Embedding(bigram, d) if bigram else None  # bigram hash gömmesi (rekor 62), sıfırdan başlar
        hd = d // self.h
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(4096).float()
        fr = torch.outer(t, inv)
        self.register_buffer("cos", fr.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", fr.sin()[None, :, None, :], persistent=False)
        self.tg = nn.Embedding(trigram, d) if trigram else None  # overtok: 3-gram hash gömmesi, sıfırdan başlar
        self.hub_k = hub
        self.hub = nn.Parameter(torch.zeros(hub, d)) if hub else None  # hub4: öğrenilmiş baş jetonlar
        self.x0 = x0

    def govde(self, x, causal=True):
        T = x.size(1); cos, sin = self.cos[:, :T], self.sin[:, :T]
        h = self.emb(x)
        if self.bg is not None:  # (önceki, şimdiki) token çifti → hash satırı; dizi başında önceki = 0
            onceki = F.pad(x[:, :-1], (1, 0))
            h = h + self.bg((onceki * V + x) % self.bg.num_embeddings)
        if self.tg is not None:  # (iki önceki, önceki, şimdiki) → hash satırı
            o1, o2 = F.pad(x[:, :-1], (1, 0)), F.pad(x[:, :-2], (2, 0))
            h = h + self.tg(((o2 * V + o1) * V + x) % self.tg.num_embeddings)
        if self.hub is not None:
            h = torch.cat([self.hub.unsqueeze(0).expand(x.size(0), -1, -1).to(h.dtype), h], 1)
            T = h.size(1); cos, sin = self.cos[:, :T], self.sin[:, :T]
        x0 = h if self.x0 else None
        for _ in range(self.tekrar):  # dongu2: aynı bloklar iki kez (ağırlık paylaşımı)
            for b in self.bloklar:
                h = b(h, cos, sin, x0, x, causal, self.hub_k)
        h = self.nf(h)
        return h[:, self.hub_k:] if self.hub is not None else h

    def aux_top(self):
        a = [b.aux for b in self.bloklar if b.aux is not None]
        return sum(a) if a else None

    def forward(self, x, causal=True):
        h = self.govde(x, causal)
        lo, lo2 = self.bas(h), (self.bas2(h) if self.bas2 is not None else None)
        if self.softcap:  # logit yumuşak tavanı (rekor 9/18)
            lo = self.softcap * torch.tanh(lo / self.softcap)
            lo2 = self.softcap * torch.tanh(lo2 / self.softcap) if lo2 is not None else None
        return lo, lo2


class LSTMLM(nn.Module):
    def __init__(self, L, d):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.rnn = nn.LSTM(d, d, num_layers=L, batch_first=True)
        self.bas = nn.Linear(d, V, bias=False); self.bas2 = None
        self.kip = LSTM_KIP
        if self.kip == "v2":  # gömme 0,02 ölçekle başlıyor; normsuz LSTM'e minik girdi gidiyor -> unigram düzlüğü
            self.n_gir = RMSNorm(d); self.n_cik = RMSNorm(d)
        with torch.no_grad():  # unutma kapısı sapması 1 (Jozefowicz 2015): düzlükten çıkış
            for k in range(L):
                getattr(self.rnn, f"bias_ih_l{k}")[d:2 * d].fill_(1.0); getattr(self.rnn, f"bias_hh_l{k}")[d:2 * d].zero_()

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):  # LSTM tam hassasiyet (bf16'da küçük gradyanlar kayboluyor)
            e = self.emb(x).float()
            if self.kip == "v2":
                e = self.n_gir(e)
            h, _ = self.rnn(e)
            if self.kip == "v2":
                h = self.n_cik(h)
            return self.bas(h), None


def ns5(G, steps=5):
    """Newton–Schulz ile yaklaşık dik matris (Muon; Keller Jordan 2024)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    tr = G.size(-2) > G.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, wd=0.0, rms=False):
        # wd + rms: Moonlight ("Muon is Scalable", 2025) — ayrık ağırlık çürümesi ve güncelleme RMS'ini AdamW'ye eşleme
        # (0,2·√max(satır, sütun)); böylece AdamW'nin LR'si ve şeması aynen kullanılır (muonw kolu, Ek 5)
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd, rms=rms))

    @torch.no_grad()
    def step(self):
        for g_ in self.param_groups:
            for p in g_["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g_["momentum"])
                g = p.grad.lerp(buf, g_["momentum"])  # Nesterov
                g = ns5(g, g_["ns_steps"])
                if g_["wd"]:
                    p.mul_(1 - g_["lr"] * g_["wd"])
                olc = 0.2 * max(p.size(-2), p.size(-1)) ** 0.5 if g_["rms"] else max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g.type_as(p), alpha=-g_["lr"] * olc)


class Hyperball(torch.optim.Optimizer):
    """muonh (A16; Wen ve ark. 2026, arXiv 2606.16899 özetinden; EVREN_TEK_ATIS\\egit_h.py ile aynı matematik):
    R = ilk adımdaki ‖W‖_F; u = NS5(Nesterov momentum); W ← W − η·R·u/‖u‖_F; W ← R·W/‖W‖_F (ağırlık azaltma yok)."""

    def __init__(self, params, lr=0.008, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g_ in self.param_groups:
            for p in g_["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                    st["R"] = p.norm().clamp_min(1e-8).clone()
                buf, R = st["buf"], st["R"]
                buf.lerp_(p.grad, 1 - g_["momentum"])
                u = ns5(p.grad.lerp(buf, g_["momentum"]), g_["ns_steps"]).type_as(p)
                p.add_(u, alpha=-(g_["lr"] * R / u.norm().clamp_min(1e-8)).item())
                p.mul_(R / p.norm().clamp_min(1e-8))


def opt_kur(model, kol, lr_tepe):
    """[(optimizer, temel LR)]; temel LR şema çarpanıyla ölçeklenir."""
    k = set(kol.split("+"))
    if not ({"muon", "muonw", "muonh"} & k):
        return [(torch.optim.AdamW(model.parameters(), lr=lr_tepe, betas=(0.9, 0.95), weight_decay=AW_WD, fused=True), lr_tepe)]
    secim = lambda n, p: (p.dim() == 2 or (p.dim() == 3 and ".moe." in n)) and n.startswith("bloklar") and ".router." not in n and ".ve." not in n  # noqa: E731
    mu = [p for n, p in model.named_parameters() if secim(n, p)]
    ad = [p for n, p in model.named_parameters() if not secim(n, p)]
    if "muonh" in k:
        assert not ({"muon", "muonw"} & k), "muonh, muon/muonw ile birlikte kullanılmaz"
        ilk = (Hyperball(mu, lr=H_TEPE), H_TEPE)
    else:
        ilk = (Muon(mu, lr=lr_tepe, wd=0.1, rms=True), lr_tepe) if "muonw" in k else (Muon(mu, lr=MUON_LR), MUON_LR)
    return [ilk,
            (torch.optim.AdamW(ad, lr=lr_tepe, betas=(0.9, 0.95), weight_decay=AW_WD, fused=True), lr_tepe)]


def buyut(model, L, optlar, kol, lr_tepe):
    """buyuyen: her blok ardışık kopyalanır (tek L'de fazlası sondan atılır); optimizer durumları (AdamW momentleri,
    Muon tamponu) kopyalara da kopyalanır. Birleşik kollarda (ör. muon+buyuyen) bütün optimizerler yeniden kurulur."""
    eski = list(model.bloklar); yeni = []; kaynak = {}
    for b in eski:
        b.aux = None  # moe8: son ileri geçişin yük dengeleme tensörü kopyalanamaz (deepcopy); hesabı etkilemez
    for b in eski:
        yeni.append(b)
        if len(yeni) < L:
            k = copy.deepcopy(b); yeni.append(k)
            for pn, ps in zip(k.parameters(), b.parameters()):
                kaynak[pn] = ps
    model.bloklar = nn.ModuleList(yeni[:L])
    eski_durum = {}
    for o, _ in optlar:
        for p, st in o.state.items():
            eski_durum[p] = st
    yeni_opt = opt_kur(model, kol, lr_tepe)
    for o, _ in yeni_opt:
        for g in o.param_groups:
            for p in g["params"]:
                src = kaynak.get(p, p)
                if src in eski_durum:
                    o.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in eski_durum[src].items()}
    return yeni_opt


def bigram_tablo(tr, n):
    """ngram: modelin göreceği token akışındaki (ilk n token) bigram olasılıkları, düzleştirme 0,01/V."""
    say = torch.zeros(V * V, dtype=torch.float32, device="cuda")
    P = 16_000_000
    for i in range(0, n - 1, P):
        a = torch.from_numpy(np.asarray(tr[i:min(n, i + P + 1)]).astype(np.int64)).cuda()
        say += torch.bincount(a[:-1] * V + a[1:], minlength=V * V).float()
        del a
    say = say.view(V, V) + 0.01 / V
    q = (say / say.sum(1, keepdim=True)).to(torch.bfloat16)
    del say; torch.cuda.empty_cache()
    return q


def flop_hesap(arch, kol, N, d, adim, N_yarim=None):
    """Eğitim hesabı ≈ 6 · (gövde + çıkış başı) · token. dongu2 gövdeyi iki kez geçer; mtp2'nin ikinci başı var;
    buyuyen ilk yarıda yarım gövdeyle eğitilir. Gömme (tablo okuma) sayılmaz. katsayi.py aynı formülü kullanır."""
    if arch != "gpt":
        return 6.0 * N * adim * TOK_ADIM
    k = set(kol.split("+"))
    bas = V * d * (2 if "mtp2" in k else 1)
    g = N * (2 if "dongu2" in k else 1)
    if "buyuyen" in k and N_yarim:
        y = adim // 2
        return 6.0 * TOK_ADIM * (y * (N_yarim + bas) + (adim - y) * (N + bas))
    return 6.0 * (g + bas) * adim * TOK_ADIM


def govde_param(m):
    return sum(p.numel() for n, p in m.named_parameters() if not (n.startswith("emb") or n.startswith("bas") or n.startswith("bg")
                                                                 or n.startswith("tg") or n == "hub" or ".ve." in n))


def init_(m, arch, olcek, L, sifir=False):
    for n, p in m.named_parameters():
        if n.startswith("bg.") or n.startswith("tg.") or ".ve." in n:  # hash/değer gömme tabloları sıfırdan başlar (başta etkisiz)
            nn.init.zeros_(p); continue
        if ".cA." in n or ".cC." in n:
            nn.init.normal_(p, 0.0, 0.02); continue
        if p.dim() < 2:
            continue
        if arch == "gpt":
            cik = n.endswith("o.weight") or n.endswith("f2.weight") or n.endswith("moe.w2")
            if sifir and cik:  # sıfır başlangıçlı çıkış projeksiyonları (rekor 5, muP benzeri)
                nn.init.zeros_(p); continue
            std = 0.02 / math.sqrt(2 * L) if cik else 0.02
            nn.init.normal_(p, 0.0, std * olcek)
        elif n.startswith("emb") or n.startswith("bas"):
            nn.init.normal_(p, 0.0, 0.02 * olcek)
        else:
            nn.init.uniform_(p, -olcek / math.sqrt(p.size(-1)), olcek / math.sqrt(p.size(-1)))


def veri():
    tr = np.memmap(os.path.join(MD, "train.bin"), dtype=np.uint16, mode="r")
    va = np.memmap(os.path.join(MD, "val.bin"), dtype=np.uint16, mode="r")
    return tr, va


def sira_dizini(tr, kip, gerek):
    """veri_mufredat: tabanın göreceği AYNI belge kümesini (doğal sıradaki ilk ~gerek token) kolaydan zora dizer.
    Zorluk = genel unigram'a göre ortalama sürpriz, kısa belgeler için medyana büzülmüş (k=50). None = doğal sıra."""
    if kip not in ("veri_mufredat", "veri_budama"):
        return None
    st = np.load(os.path.join(MD, "train_doc_starts.npy"))
    sk = np.load(os.path.join(MD, "mufredat_skor.npy")).astype(np.float64)
    uz = np.diff(np.append(st, len(tr)))
    havuz = 2.0 if kip == "veri_budama" else 1.0  # budama: 2 kat havuzdan orta %50 (oran her boyutta sabit)
    k = int(np.searchsorted(st, int(gerek * 1.02 * havuz) + 4096)) + 1
    k = min(k, len(st))
    st, sk, uz = st[:k], sk[:k], uz[:k]
    med = float(np.median(sk))
    buz = (sk * np.maximum(uz - 1, 0) + 50 * med) / (np.maximum(uz - 1, 0) + 50)
    if kip == "veri_budama":
        lo, hi = np.percentile(buz, [25, 75])
        tut = np.where((buz >= lo) & (buz <= hi))[0]  # doğal sırada kalır (müfredat karışmasın)
        assert uz[tut].sum() >= gerek, "budama havuzu yetmedi"
        return np.stack([st[tut], uz[tut]], 1)
    o = np.argsort(buz, kind="stable")
    return np.stack([st[o], uz[o]], 1)


class Akis:
    """Sabit sıralı token akışı; her çağrıda B satır x (T+1) token verir. sarma: ep4 için benzersiz veri sınırı (başa döner);
    son: kesin üst sınır (rholoss: referansın verisine girilmez)."""

    def __init__(self, tr, sira, sarma=None, son=None):
        self.tr, self.pos, self.sira = tr, 0, sira
        self.sarma, self.son = sarma, son
        if sira is not None:
            self.bi, self.boff, self.tampon = 0, 0, np.zeros(0, dtype=np.uint16)

    def al(self, n):
        if self.sira is None:
            if self.sarma is not None and self.pos + n > self.sarma:
                self.pos = 0
            if self.son is not None:
                assert self.pos + n <= self.son, "veri bitti (rholoss: referans bölgesine girilmez)"
            out = np.asarray(self.tr[self.pos:self.pos + n]); self.pos += n - 1  # bir token örtüşme: hedef kayması
            assert len(out) == n, "veri bitti"
            return out
        parca = [self.tampon]; topl = len(self.tampon)
        while topl < n:
            s, l = self.sira[self.bi]; self.bi += 1
            parca.append(np.asarray(self.tr[s:s + l])); topl += l
        buf = np.concatenate(parca)
        self.tampon = buf[n - 1:]
        return buf[:n]

    def parti(self, B, T):
        a = torch.from_numpy(self.al(B * T + 1).astype(np.int64))
        x = a[:-1].view(B, T); y = a[1:].view(B, T)
        return x.pin_memory().cuda(non_blocking=True), y.pin_memory().cuda(non_blocking=True)

    def durum(self):
        d = {"pos": self.pos}
        if self.sira is not None:
            d.update(bi=self.bi, boff=self.boff, tampon=self.tampon.copy())
        return d

    def yukle(self, d):
        self.pos = d["pos"]
        if self.sira is not None:
            self.bi, self.boff, self.tampon = d["bi"], d["boff"], d["tampon"]


@torch.no_grad()
def degerlendir(model, va, tb, n_tok=2_000_000):
    model.eval()
    T, B = T_SON, 16
    nat, byt, ntok = 0.0, 0.0, 0
    i = 0
    while ntok < n_tok and i + B * T + 1 <= len(va):
        a = torch.from_numpy(np.asarray(va[i:i + B * T + 1]).astype(np.int64)).cuda()
        x, y = a[:-1].view(B, T), a[1:].view(B, T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, _ = model(x)
        l = F.cross_entropy(lo.float().view(-1, V), y.reshape(-1), reduction="none")
        nb = tb[y.reshape(-1)]
        m = nb > 0
        nat += (l * m).sum().item(); byt += nb.sum().item(); ntok += int(m.sum().item())
        i += B * T
    model.train()
    return nat / max(ntok, 1), nat / (math.log(2) * max(byt, 1))


def ref_egit(tr, tohum):
    """rholoss referansı: g1 taban (AdamW, ısınma %2 + kosinüs), train.bin'in son RHO_HOLDOUT jetonundan 20 jeton/parametre.
    Ana eğitim bu bölgeye girmez. Döner: (model, eğitim hesabı)."""
    L, d = GPT_BOY["g1"]
    torch.manual_seed(10_000 + tohum)
    ref = GPT(L, d).cuda(); init_(ref, "gpt", 1.0, L)
    N = govde_param(ref); adim = max(50, int(20 * N) // TOK_ADIM)
    if int(os.environ.get("MERDIVEN_ADIM_MAX", "0")):  # yalnız duman sınaması
        adim = min(adim, int(os.environ["MERDIVEN_ADIM_MAX"]))
    lr = 1.5e-3 * 256 / d; isinma = max(20, int(0.02 * adim))
    opt = torch.optim.AdamW(ref.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    ak = Akis(tr, None); ak.pos = len(tr) - RHO_HOLDOUT
    Bm, nmik = MIKRO // T_SON, TOK_ADIM // MIKRO
    for s in range(adim):
        p_ = (s - isinma) / max(1, adim - isinma)
        for g in opt.param_groups:
            g["lr"] = lr * (s + 1) / isinma if s < isinma else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p_)))
        for _ in range(nmik):
            x, y = ak.parti(Bm, T_SON)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, _ = ref(x)
            F.cross_entropy(lo.float().view(-1, V), y.reshape(-1)).div(nmik).backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref, 6.0 * (N + V * d) * adim * TOK_ADIM


def _kosu_kimligi(arch, boy, kol, tohum, tpp, lr_carp, sonuc, etiket):
    ortam = {k: os.environ.get(k, "") for k in ("MERDIVEN_WD", "MERDIVEN_H_TEPE", "MERDIVEN_MUON_LR", "MERDIVEN_ADIM_MAX",
                                                "MERDIVEN_MIKRO", "MERDIVEN_LSTM", "MERDIVEN_DERLE")}
    anahtar = json.dumps([arch, boy, kol, tohum, tpp, lr_carp, os.path.basename(sonuc), etiket, ortam], sort_keys=True)
    return f"{boy}_{hashlib.sha1(anahtar.encode()).hexdigest()[:16]}"


def egit(arch, boy, kol, tohum, tpp, lr_carp, sonuc, etiket=""):
    kollar = set(kol.split("+"))  # birleşik kol: ör. "muon+canon+ngram" (Ek 3)
    L, d = (GPT_DERIN if "derin" in kollar else GPT_BOY)[boy] if arch == "gpt" else LSTM_BOY[boy]
    torch.manual_seed(tohum); np.random.seed(tohum)
    if CUDA_VAR:
        torch.cuda.reset_peak_memory_stats()
    model = (GPT(L, d, mtp=("mtp2" in kollar), canon=("canon" in kollar), tekrar=(2 if "dongu2" in kollar else 1),
                 qknorm=("qknorm" in kollar), relu2=("relu2" in kollar), softcap=(SOFTCAP if "softcap" in kollar else 0.0),
                 bigram=(BIGRAM_M if ({"bigram", "overtok"} & kollar) else 0), trigram=(TRIGRAM_M if "overtok" in kollar else 0),
                 hub=(HUB_K if "hub4" in kollar else 0), ve=("ve" in kollar), x0=("x0" in kollar),
                 moe=(MOE_E if "moe8" in kollar else 0), maske=("gptbert" in kollar))
             if arch == "gpt" else LSTMLM(L, d)).cuda()
    duz = "duz_dusuk" in kollar
    init_(model, arch, 0.3 if duz else 1.0, L, sifir=("sifir" in kollar))
    N = govde_param(model)  # buyuyen için de son (tam) modelin gövdesi
    N_aktif = N - L * (MOE_E - MOE_K) * 4 * d * d if "moe8" in kollar else N  # moe8: jeton başına çalışan uzmanlar
    D = int(tpp * N)
    adim = max(50, D // TOK_ADIM)
    if int(os.environ.get("MERDIVEN_ADIM_MAX", "0")):  # yalnız duman sınaması
        adim = min(adim, int(os.environ["MERDIVEN_ADIM_MAX"]))
    N_yarim = None
    if "buyuyen" in kollar:
        model.bloklar = nn.ModuleList(list(model.bloklar)[:(L + 1) // 2])
        N_yarim = govde_param(model)
    lr_tepe = (1.5e-3 * 256 / d if arch == "gpt" else 3e-3 * math.sqrt(256 / d)) * lr_carp
    isinma = max(20, int(0.02 * adim))
    optlar = opt_kur(model, kol, lr_tepe)
    ileri = torch.compile(model) if (arch == "gpt" and DERLE) else model
    tr, va = veri()
    veri_kip = "veri_budama" if "veri_budama" in kollar else ("veri_mufredat" if "veri_mufredat" in kollar else "taban")
    gerek = adim * (TOK_ADIM + TOK_ADIM // MIKRO)
    akis = Akis(tr, sira_dizini(tr, veri_kip, gerek), sarma=(gerek // 4 + T_SON + 2 if "ep4" in kollar else None),
                son=(len(tr) - RHO_HOLDOUT if "rholoss" in kollar else None))
    tb = torch.load(os.path.join(CACHE, "tokenizer", "token_bytes.pt")).cuda()
    bigram = bigram_tablo(tr, adim * TOK_ADIM + 1) if "ngram" in kollar else None
    rho = "rholoss" in kollar
    ref, ref_flop = (None, 0.0)

    def lr_at(s):
        if duz:
            return 0.1 * lr_tepe
        if s < isinma:
            return lr_tepe * (s + 1) / isinma
        if "wsd" in kollar:  # sabit, son %45'te doğrusal iniş tabanın tabanı %10'a (rekor 26: soğuma payı 0,45)
            soguma = max(1, int(0.45 * adim))
            if s < adim - soguma:
                return lr_tepe
            return lr_tepe * (1 - 0.9 * (s - (adim - soguma)) / soguma)
        p = (s - isinma) / max(1, adim - isinma)
        return lr_tepe * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def T_at(s):
        if "seq_mufredat" not in kollar:
            return T_SON
        return [64, 128, 256, 512][min(3, int(4 * s / adim))]

    # ---- checkpoint / devam (kuyruk6; MERDIVEN_CKPT verilmezse kapalı, eski davranış) ----
    ck = os.path.join(CKPT_DIR, _kosu_kimligi(arch, boy, kol, tohum, tpp, lr_carp, sonuc, etiket) + ".pt") if CKPT_DIR else None
    if ck:
        os.makedirs(CKPT_DIR, exist_ok=True)
    s0, gecen, kayip_iz, devam_sayisi = 0, 0.0, [], 0
    if ck and os.path.exists(ck):
        k = torch.load(ck, map_location="cuda" if CUDA_VAR else "cpu", weights_only=False)
        if "buyuyen" in kollar and k["s"] > adim // 2:  # kayıt büyümeden sonra: önce yapıyı büyüt, sonra durumları yükle
            optlar = buyut(model, L, optlar, kol, lr_tepe); ileri = torch.compile(model) if DERLE else model
        model.load_state_dict(k["model"])
        for (o, _), st in zip(optlar, k["optlar"]):
            o.load_state_dict(st)
        akis.yukle(k["akis"])
        torch.set_rng_state(k["rng"]); np.random.set_state(k["np_rng"])
        if CUDA_VAR and k.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(k["cuda_rng"])
        s0, gecen, kayip_iz, devam_sayisi = k["s"], k["gecen"], k["kayip_iz"], k.get("devam", 0) + 1
        if rho:
            ref, ref_flop = k["ref"], k["ref_flop"]
        print(f"[{boy}] DEVAM: adım {s0}/{adim} (checkpoint {os.path.basename(ck)})", flush=True)
    if rho and ref is None:
        ref, ref_flop = ref_egit(tr, tohum)
        print(f"[{boy}] rholoss referansı hazır (hesap {ref_flop:.3e})", flush=True)

    def kaydet(s_sonraki):
        k = dict(model=model.state_dict(), optlar=[o.state_dict() for o, _ in optlar], s=s_sonraki, akis=akis.durum(),
                 rng=torch.get_rng_state(), np_rng=np.random.get_state(),
                 cuda_rng=(torch.cuda.get_rng_state() if CUDA_VAR else None),
                 gecen=gecen + (time.time() - t0), kayip_iz=list(kayip_iz), devam=devam_sayisi, ref=ref, ref_flop=ref_flop,
                 kol=kol, boy=boy, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(k, ck + ".tmp"); os.replace(ck + ".tmp", ck)

    t0 = time.time(); son_kayit = time.time()
    for s in range(s0, adim):
        if "buyuyen" in kollar and s == adim // 2:
            optlar = buyut(model, L, optlar, kol, lr_tepe); ileri = torch.compile(model) if DERLE else model
            print(f"[{boy}] BUYUDU adim {s}: {len(model.bloklar)} blok", flush=True)
        for o, b0 in optlar:
            for g in o.param_groups:
                g["lr"] = lr_at(s) if b0 == lr_tepe else b0 * lr_at(s) / lr_tepe
        T = T_at(s); Bm = MIKRO // T; nmik = TOK_ADIM // MIKRO
        top, n_top = 0.0, 0
        for j in range(nmik):
            if rho:  # öğrenilebilir örneğe öncelik: 2 kat aday satırdan (eğitim − referans kaybı) en yüksek yarısı
                xa, ya = akis.parti(2 * Bm, T)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lm, _ = ileri(xa); lr_, _ = ref(xa)
                cm = F.cross_entropy(lm.float().view(-1, V), ya.reshape(-1), reduction="none").view(2 * Bm, T).mean(1)
                cr = F.cross_entropy(lr_.float().view(-1, V), ya.reshape(-1), reduction="none").view(2 * Bm, T).mean(1)
                sec = torch.topk(cm - cr, Bm).indices
                x, y = xa[sec], ya[sec]
                del lm, lr_
            else:
                x, y = akis.parti(Bm, T)
            if "gptbert" in kollar and j % GB_ORAN == GB_ORAN - 1:  # maskeli bir sonraki jeton tahmini (çift yönlü)
                mk = torch.rand(x.shape, device=x.device) < GB_MASKE
                mk[:, 0] = False
                xm = x.masked_fill(mk, V)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo, _ = ileri(xm, causal=False)
                hm = torch.zeros_like(mk); hm[:, :-1] = mk[:, 1:]
                lm_ = F.cross_entropy(lo.float()[hm], y[hm]) if hm.any() else lo.float().sum() * 0.0
                aux = model.aux_top() if "moe8" in kollar else None
                (lm_ + (MOE_AUX * aux if aux is not None else 0.0)).div(nmik).backward()
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, lo2 = ileri(x)
            if bigram is not None and s < adim // 4:  # ngram: ilk %25'te hedefe bigram karışır, λ 0,5 -> 0
                lam = 0.5 * (1 - s / (adim // 4))
                lp = F.log_softmax(lo.float().view(-1, V), -1)
                ce = F.nll_loss(lp, y.reshape(-1))
                yum = -(bigram[x.reshape(-1)].float() * lp).sum(-1).mean()
                ((1 - lam) * ce + lam * yum).div(nmik).backward()
                top += ce.item() / nmik
                continue
            if "zpd" in kollar:  # yakınsal gelişim: p≈0,5 tokenlara ağırlık, taban 0,1 (ölçü hep düz CE)
                ce = F.cross_entropy(lo.float().view(-1, V), y.reshape(-1), reduction="none")
                with torch.no_grad():
                    pr = torch.exp(-ce); w = (4 * pr * (1 - pr)).clamp_min(0.1)
                loss = ce.mean()
                ((w * ce).sum() / w.sum()).div(nmik).backward()
                top += loss.item() / nmik
                continue
            loss = F.cross_entropy(lo.float().view(-1, V), y.reshape(-1))
            ek = MOE_AUX * model.aux_top() if "moe8" in kollar else None
            if lo2 is not None:
                loss2 = F.cross_entropy(lo2[:, :-1].float().reshape(-1, V), y[:, 1:].reshape(-1))
                (loss + 0.5 * loss2).div(nmik).backward()
            elif ek is not None:
                (loss + ek).div(nmik).backward()
            else:
                loss.div(nmik).backward()
            if "gptbert" in kollar:
                top += loss.item(); n_top += 1
            else:
                top += loss.item() / nmik
        if n_top:
            top /= n_top
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, _ in optlar:
            o.step(); o.zero_grad(set_to_none=True)
        if not math.isfinite(top):
            print(f"[{boy}] KAYIP PATLADI adim {s}", flush=True); break
        if s % max(1, adim // 20) == 0 or s == adim - 1:
            kayip_iz.append((s, round(top, 4)))
            print(f"[{arch}/{boy}/{kol}/t{tohum}] adim {s}/{adim} kayip {top:.4f} lr {lr_at(s):.2e} T {T} "
                  f"{(s + 1 - s0) * TOK_ADIM / (time.time() - t0):,.0f} tok/sn", flush=True)
        if ck and s < adim - 1:
            if SINA_DUR_ADIM is not None and s + 1 == SINA_DUR_ADIM and DUR:
                open(DUR, "w").write("sina")
            if DUR and os.path.exists(DUR):
                kaydet(s + 1)
                print(f"[{boy}] DURAKLATILDI adım {s + 1}/{adim} (checkpoint {os.path.basename(ck)})", flush=True)
                sys.exit(3)
            if time.time() - son_kayit > CKPT_DK * 60:
                kaydet(s + 1); son_kayit = time.time()
    sure = gecen + (time.time() - t0)
    vl, vbpb = degerlendir(model, va, tb)
    N_yarim_f = (N_yarim - ((L + 1) // 2) * (MOE_E - MOE_K) * 4 * d * d) if (N_yarim and "moe8" in kollar) else N_yarim
    fl = flop_hesap(arch, kol, N_aktif, d, adim, N_yarim_f)
    if "hub4" in kollar:
        fl *= (T_SON + HUB_K) / T_SON
    if rho:  # aday satırların ileri geçişi (2 kat) + referansın ileri geçişi + referansın eğitimi
        Lr, dr = GPT_BOY["g1"]
        fl += 2.0 * (N_aktif + V * d) * 2 * adim * TOK_ADIM + 2.0 * (govde_param(ref) + V * dr) * 2 * adim * TOK_ADIM + ref_flop
    satir = dict(arch=arch, boy=boy, kol=kol, tohum=tohum, L=L, d=d, N=N,
                 N_toplam=sum(p.numel() for p in model.parameters()), tokens=adim * TOK_ADIM, adim=adim,
                 lr_tepe=lr_tepe, tpp=tpp, val_nat=round(vl, 5), val_bpb=round(vbpb, 5), egitim_sn=round(sure, 1),
                 tok_sn=round(adim * TOK_ADIM / sure), kayip_iz=kayip_iz, etiket=etiket,
                 alet=(LSTM_KIP if arch == "lstm" else "v1"),
                 flop=fl, N_yarim=N_yarim,
                 muon_lr=(MUON_LR if "muon" in kollar else ("lr_tepe+wd0.1+rms" if "muonw" in kollar else None)), derle=(DERLE if arch == "gpt" else None),
                 aw_wd=AW_WD, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
    if "muonh" in kollar:
        satir["h_tepe"] = H_TEPE
    if "moe8" in kollar:
        satir["N_aktif"] = N_aktif
    if CUDA_VAR:
        satir["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if devam_sayisi:
        satir["devam"] = devam_sayisi
    with open(sonuc, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(satir, ensure_ascii=False) + "\n")
    print("SONUC", json.dumps({k: satir[k] for k in ("arch", "boy", "kol", "tohum", "N", "tokens", "val_nat", "val_bpb", "egitim_sn")}), flush=True)
    if ck and os.path.exists(ck):
        os.remove(ck)
    del model, optlar, ileri, bigram
    if CUDA_VAR:
        torch.cuda.empty_cache()
    return satir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpt", choices=["gpt", "lstm"])
    ap.add_argument("--sizes", default="g1,g2,g3,g4")
    ap.add_argument("--variant", default="taban")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tpp", type=float, default=20.0)
    ap.add_argument("--lr_mult", type=float, default=1.0)
    ap.add_argument("--out", default="sonuclar.jsonl")
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    for b in a.sizes.split(","):
        egit(a.arch, b, a.variant, a.seed, a.tpp, a.lr_mult, a.out, a.etiket)
    print("MERDIVEN BITTI", flush=True)
