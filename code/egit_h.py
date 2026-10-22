# -*- coding: utf-8 -*-
"""egit_evren.py + iki yeni kol (8 Ekim 2026 gece; ön kayıt ON_KAYIT_BUYUK_MERDIVEN_2026-10-08.md):
  muonh    — Hyperball (Wen, Dang, Lyu, Ma, Liang 2026, arXiv 2606.16899; A16)
  buyuyen  — model büyütme / Untied-Grow (Chen, Vegesna, Dahal, Wilson 2026, arXiv 2609.19107; A17): yarım derinlikle başla,
             adımların yarısında her blok ardışık kopyalanır (merdiven.buyut; optimizer durumları kopyalara da kopyalanır).
egit_evren.py ve merdiven.py'ye DOKUNMAZ (kanıt koşusu onları kullanıyor); ikisini içe aktarır. Eğitim döngüsü egit_evren.main'in
kopyasıdır, yalnız (1) optimizer kurulumu muonh için, (2) büyüme kancası, (3) büyümede hesap (N_yarim) farkları eklendi.
Öteki kollar egit_evren ile aynı davranır.

Hyperball (bu uygulama; makalenin gövdesi okunmadı, özet: "ağırlık matrislerinin ve güncellemelerinin Frobenius normu sabit"):
  gövde 2B ağırlıkları için R = ilk adımdaki ‖W‖_F; u = NS5(Nesterov momentum gradyanı);
  W ← W − η·R·u/‖u‖_F;  W ← R·W/‖W‖_F   (ağırlık azaltma YOK). η = --h_tepe, egit_evren LR şemasıyla aynı oranda ölçeklenir.
Kullanım: python egit_h.py --L 8 --d 512 --kol canon+muonh+qknorm+bigram+wsd --h_tepe 0.008 ... (egit_evren argümanları geçerli)"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import egit_evren as E  # noqa: E402
M = E.M

H_TEPE = 0.008
for _i, _x in enumerate(list(sys.argv)):
    if _x == "--h_tepe":
        H_TEPE = float(sys.argv[_i + 1]); del sys.argv[_i:_i + 2]; break
DESTEK = set(E.DESTEK) | {"muonh", "buyuyen"}


class Hyperball(torch.optim.Optimizer):
    """Muon yönü + sabit Frobenius normlu ağırlık ve güncelleme. 'lr' burada göreli adım boyudur (η)."""
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
                u = M.ns5(p.grad.lerp(buf, g_["momentum"]), g_["ns_steps"]).type_as(p)
                p.add_(u, alpha=-(g_["lr"] * R / u.norm().clamp_min(1e-8)).item())
                p.mul_(R / p.norm().clamp_min(1e-8))


_eski_opt_kur = M.opt_kur


def opt_kur_h(model, kol, lr_tepe):
    k = set(kol.split("+"))
    if "muonh" not in k:
        return _eski_opt_kur(model, kol, lr_tepe)
    assert not ({"muon", "muonw"} & k), "muonh, muon/muonw ile birlikte kullanılmaz"
    mu = [p for n, p in model.named_parameters() if p.dim() == 2 and n.startswith("bloklar")]
    ad = [p for n, p in model.named_parameters() if not (p.dim() == 2 and n.startswith("bloklar"))]
    # temel LR = H_TEPE: döngü bunu b0 * lr_at(s) / lr_tepe ile şemaya göre ölçekler
    return [(Hyperball(mu, lr=H_TEPE), H_TEPE),
            (torch.optim.AdamW(ad, lr=lr_tepe, betas=(0.9, 0.95), weight_decay=0.1, fused=torch.cuda.is_available()), lr_tepe)]


M.opt_kur = opt_kur_h  # merdiven.buyut da bunu kullanır (modül küreseli)


def main():
    a = E.ayrıştır()
    kollar = set(a.kol.split("+"))
    assert kollar <= DESTEK, f"bu kodda olmayan kol: {kollar - DESTEK}"
    W, R_, cihaz = E.kur_dagitik(a)
    ana = R_ == 0
    torch.manual_seed(a.tohum); np.random.seed(a.tohum)
    model = E.model_kur(a, kollar).to(cihaz)
    N = M.govde_param(model)  # büyüyen için de son (tam) modelin gövdesi
    adim = a.adim or int(a.token // a.tok_adim)
    assert adim > 0 and a.tok_adim % (a.mikro * W) == 0, "tok_adim, mikro × süreç sayısına tam bölünmeli"
    nmik = a.tok_adim // (a.mikro * W)
    lr_tepe = 1.5e-3 * 256 / a.d * a.lr_carp
    M.MUON_LR = a.muon_lr
    buyu = "buyuyen" in kollar
    s_buyu = adim // 2
    N_yarim = None
    if buyu:
        model.bloklar = nn.ModuleList(list(model.bloklar)[:(a.L + 1) // 2])
        N_yarim = M.govde_param(model)
    optlar = M.opt_kur(model, a.kol, lr_tepe)
    isinma = max(20, int(0.02 * adim))

    def sar():
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[cihaz.index] if a.cihaz == "cuda" else None) if W > 1 else \
            (torch.compile(model) if E.DERLE else model)
    sarili = sar()
    dt = np.uint16 if a.dtype == "uint16" else np.uint32
    tr = np.memmap(a.veri, dtype=dt, mode="r"); va = np.memmap(a.dogrulama, dtype=dt, mode="r")
    os.makedirs(a.kayit, exist_ok=True)
    yol = os.path.join(a.kayit, "son.pt")
    s0, pos = 0, 0
    if a.devam and os.path.exists(yol):
        k = torch.load(yol, map_location=cihaz, weights_only=False)
        if buyu and k["adim"] > s_buyu:  # kayıt büyümeden sonra: önce yapıyı büyüt, sonra durumları yükle
            optlar = M.buyut(model, a.L, optlar, a.kol, lr_tepe); sarili = sar()
        model.load_state_dict(k["model"])
        for (o, _), st in zip(optlar, k["optlar"]):
            o.load_state_dict(st)
        s0, pos = k["adim"], k["pos"]
        torch.set_rng_state(k["rng"].cpu())  # 13 Ekim: map_location=cuda RNG'yi GPU'ya taşıyordu
        if ana:
            print(f"DEVAM: {s0}. adımdan, veri konumu {pos}", flush=True)

    def lr_at(s):
        if s < isinma:
            return lr_tepe * (s + 1) / isinma
        if "wsd" in kollar:
            soguma = max(1, int(0.45 * adim))
            return lr_tepe if s < adim - soguma else lr_tepe * (1 - 0.9 * (s - (adim - soguma)) / soguma)
        p = (s - isinma) / max(1, adim - isinma)
        return lr_tepe * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def kaydet(s, pos_):
        if not ana:
            return
        k = dict(model=model.state_dict(), optlar=[o.state_dict() for o, _ in optlar], adim=s, pos=pos_, rng=torch.get_rng_state(),
                 args=vars(a), h_tepe=H_TEPE, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(k, yol + ".tmp"); os.replace(yol + ".tmp", yol)

    @torch.no_grad()
    def dogrula():
        model.eval(); T, B = a.T, 8; nat, n, i = 0.0, 0, 0
        while n < a.dogrulama_token and i + B * T + 1 <= len(va):
            x = torch.from_numpy(np.asarray(va[i:i + B * T + 1]).astype(np.int64)).to(cihaz)
            with torch.autocast(a.cihaz, dtype=torch.bfloat16, enabled=(a.cihaz == "cuda")):
                lo, _ = model(x[:-1].view(B, T))
            nat += F.cross_entropy(lo.float().reshape(-1, M.V), x[1:].reshape(-1), reduction="sum").item(); n += B * T; i += B * T
        model.train(); return nat / max(n, 1)

    gunluk = open(os.path.join(a.kayit, "gunluk.jsonl"), "a", encoding="utf-8") if ana else None
    t0 = time.time(); son_kayit = time.time(); bozuk = 0
    for s in range(s0, adim):
        if buyu and s == s_buyu and len(model.bloklar) < a.L:
            optlar = M.buyut(model, a.L, optlar, a.kol, lr_tepe); sarili = sar()
            if ana:
                print(f"BUYUDU adim {s}: {len(model.bloklar)} blok", flush=True)
                gunluk.write(json.dumps(dict(adim=s, buyudu=len(model.bloklar))) + "\n"); gunluk.flush()
        for o, b0 in optlar:
            for g in o.param_groups:
                g["lr"] = lr_at(s) if b0 == lr_tepe else b0 * lr_at(s) / lr_tepe
        if pos + a.tok_adim + 1 > len(tr):
            pos = 0
            if ana:
                print("UYARI: veri bitti, baştan alınıyor", flush=True)
        top = torch.zeros((), device=cihaz)
        for i in range(nmik):
            off = pos + (i * W + R_) * a.mikro
            buf = torch.from_numpy(np.asarray(tr[off:off + a.mikro + 1]).astype(np.int64)).to(cihaz)
            x, y = buf[:-1].view(-1, a.T), buf[1:].view(-1, a.T)
            senk = (i == nmik - 1) or W == 1
            ctx = sarili.no_sync() if (W > 1 and not senk) else torch.enable_grad()
            with ctx:
                with torch.autocast(a.cihaz, dtype=torch.bfloat16, enabled=(a.cihaz == "cuda")):
                    lo, lo2 = sarili(x)
                kayip = F.cross_entropy(lo.float().view(-1, M.V), y.reshape(-1))
                tam = kayip if lo2 is None else kayip + 0.5 * F.cross_entropy(lo2[:, :-1].float().reshape(-1, M.V), y[:, 1:].reshape(-1))
                if "moe8" in kollar:
                    tam = tam + M.MOE_AUX * model.aux_top()
                (tam / nmik).backward()
            top += kayip.detach() / nmik
        if W > 1:
            dist.all_reduce(top); top /= W
        pos += a.tok_adim
        if not torch.isfinite(top):
            bozuk += 1
            for o, _ in optlar:
                o.zero_grad(set_to_none=True)
            if ana:
                print(f"BOZUK ADIM {s} (üst üste {bozuk}) — atlandı", flush=True)
            if bozuk >= 5:
                raise SystemExit("5 bozuk adım üst üste: son kayıttan --devam ile ve daha düşük --lr_carp ile sürdür")
            continue
        bozuk = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o, _ in optlar:
            o.step(); o.zero_grad(set_to_none=True)
        if ana and (s % max(1, adim // 50) == 0 or s == adim - 1):
            sat = dict(adim=s, kayip=round(top.item(), 4), lr=lr_at(s), tok_sn=round((s - s0 + 1) * a.tok_adim / (time.time() - t0)))
            print(json.dumps(sat), flush=True); gunluk.write(json.dumps(sat) + "\n"); gunluk.flush()
        if ana and a.dogrulama_aralik and (s + 1) % a.dogrulama_aralik == 0:
            v = dogrula(); gunluk.write(json.dumps(dict(adim=s, val_nat=round(v, 5))) + "\n"); gunluk.flush()
            print(f"DOGRULAMA adim {s}: {v:.4f}", flush=True)
        zaman_geldi = (time.time() - son_kayit) / 60 >= a.kayit_dk
        if (a.kayit_adim and (s + 1) % a.kayit_adim == 0) or zaman_geldi:
            kaydet(s + 1, pos); son_kayit = time.time()
        if os.path.exists(os.path.join(a.kayit, "DUR")):
            kaydet(s + 1, pos)
            if ana:
                print(f"DURAKLATILDI: {s + 1}. adım kaydedildi (DUR dosyası)", flush=True)
            break
        if a.dur_adim and s + 1 >= a.dur_adim:
            if ana:
                print(f"DUR: {s + 1}. adımda (sınama)", flush=True)
            break
    else:
        kaydet(adim, pos)
        if ana:
            v = dogrula()
            # flop_hesap tok_adim'ı M.TOK_ADIM sayar; adım başına gerçek token a.tok_adim olduğu için ölçeklenir (egit_evren ile aynı)
            sonuc = dict(kol=a.kol, L=a.L, d=a.d, N=N, N_yarim=N_yarim, h_tepe=(H_TEPE if "muonh" in kollar else None), adim=adim,
                         token=adim * a.tok_adim, val_nat=round(v, 5),
                         flop=E.hesap_duzelt(kollar, N, a, adim, N_yarim=N_yarim),
                         sure_sn=round(time.time() - t0, 1), surec=W, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(sonuc, open(os.path.join(a.kayit, "sonuc.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("SONUC", json.dumps(sonuc), flush=True)
    if W > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
