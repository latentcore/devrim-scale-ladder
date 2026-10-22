# -*- coding: utf-8 -*-
"""EVREN tek atış eğitim kodu (8 Ekim 2026; plan: 00_PLAN.md madde 5).
merdiven.py'nin modelini ve kollarını aynen kullanır (laptopta ölçülen tarif = EVREN'de koşan tarif): muonw, muon, canon,
qknorm, relu2, sifir, softcap, bigram, wsd, mtp2, dongu2. (ngram, zpd, seq_mufredat, veri_*, buyuyen bu kodda yok.)
Tek GPU, çok GPU (torchrun + DDP, NCCL) ve CPU (gloo; laptop sınaması) çalışır. Kayıt/devam (atomik, süreli ve adımlı),
NaN koruması (bozuk adım atlanır, üst üste 5'te durur), periyodik doğrulama, jsonl günlük, bitişte sonuc.json.
Veri: token akışı .bin (uint16; --dtype uint32 da olur), sıra sabit; her adımda TOK_ADIM token, süreçlere mikro parçalar halinde.

Laptop sınaması (CPU, 2 süreç; Windows'ta torchrun libuv istediği için süreçler elle başlatılır): sina_cpu.sh
EVREN: slurm_evren.sh (EVREN'in konteyner/bölüm ayrıntıları DOĞRULANMADI; hesap açılınca düzeltilecek)."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BURA, "..", "d1_frank", "merdiven"))
import merdiven as M  # noqa: E402

DESTEK = {"taban", "muon", "muonw", "canon", "qknorm", "relu2", "sifir", "softcap", "bigram", "wsd", "mtp2", "dongu2",
          "overtok", "hub4", "ve", "x0", "moe8"}  # kuyruk6 (12 Ekim): merdiven.py'deki model düzeyi yeni kollar; varsayılan yol aynı
DERLE = os.environ.get("EGIT_DERLE", "0") == "1"  # kuyruk6 F2+: torch.compile (tek süreç; Akıllı Uygulama Denetimi kapatıldı, 11 Ekim)


def hesap_duzelt(kollar, N, a, adim, N_yarim=None):
    """sonuc.json hesabı: moe8'de çalışan uzmanlarla, hub4'te (T+4)/T ile (merdiven.py egit ile aynı)."""
    Nf = N - a.L * (M.MOE_E - M.MOE_K) * 4 * a.d * a.d if "moe8" in kollar else N
    if N_yarim is not None and "moe8" in kollar:
        N_yarim = N_yarim - ((a.L + 1) // 2) * (M.MOE_E - M.MOE_K) * 4 * a.d * a.d
    if N_yarim is not None:
        fl = M.flop_hesap("gpt", a.kol, Nf, a.d, adim, N_yarim=N_yarim)
    else:
        fl = M.flop_hesap("gpt", a.kol, Nf, a.d, adim)
    if "hub4" in kollar:
        fl *= (a.T + M.HUB_K) / a.T
    return fl * a.tok_adim / M.TOK_ADIM


def ayrıştır():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True); ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--kol", default="muonw")
    ap.add_argument("--vocab", type=int, default=M.V)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--tok_adim", type=int, default=32768, help="adım başına toplam token (bütün süreçler)")
    ap.add_argument("--mikro", type=int, default=8192, help="süreç başına mikro parti tokenı")
    ap.add_argument("--adim", type=int, default=0); ap.add_argument("--token", type=float, default=0, help="toplam token (adim yerine)")
    ap.add_argument("--lr_carp", type=float, default=1.0); ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--veri", default=os.path.join(M.MD, "train.bin")); ap.add_argument("--dogrulama", default=os.path.join(M.MD, "val.bin"))
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--kayit", default=os.path.join(BURA, "kayit")); ap.add_argument("--kayit_dk", type=float, default=20)
    ap.add_argument("--kayit_adim", type=int, default=0, help="sınama için: her N adımda kayıt")
    ap.add_argument("--dur_adim", type=int, default=0, help="sınama için: N. adımdan sonra dur (devam sınaması)")
    ap.add_argument("--dogrulama_aralik", type=int, default=0); ap.add_argument("--dogrulama_token", type=int, default=2_000_000)
    ap.add_argument("--devam", action="store_true"); ap.add_argument("--cihaz", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tohum", type=int, default=1)
    return ap.parse_args()


def kur_dagitik(a):
    W = int(os.environ.get("WORLD_SIZE", "1")); R = int(os.environ.get("RANK", "0")); LR_ = int(os.environ.get("LOCAL_RANK", "0"))
    if W > 1:
        # Windows'ta PyTorch libuv'suz derlenmiş: laptop sınamasında env:// ?use_libuv=False ile (EVREN Linux, torchrun varsayılanı)
        dist.init_process_group(backend="nccl" if a.cihaz == "cuda" else "gloo", init_method="env://?use_libuv=False" if os.name == "nt" else "env://")
    cihaz = torch.device(f"cuda:{LR_}") if a.cihaz == "cuda" else torch.device("cpu")
    if a.cihaz == "cuda":
        torch.cuda.set_device(cihaz)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    return W, R, cihaz


def model_kur(a, kollar):
    M.V = a.vocab  # merdiven.GPT ve flop_hesap bu küresel değeri kullanır
    model = M.GPT(a.L, a.d, mtp=("mtp2" in kollar), canon=("canon" in kollar), tekrar=(2 if "dongu2" in kollar else 1),
                  qknorm=("qknorm" in kollar), relu2=("relu2" in kollar), softcap=(M.SOFTCAP if "softcap" in kollar else 0.0),
                  bigram=(M.BIGRAM_M if ({"bigram", "overtok"} & kollar) else 0), trigram=(M.TRIGRAM_M if "overtok" in kollar else 0),
                  hub=(M.HUB_K if "hub4" in kollar else 0), ve=("ve" in kollar), x0=("x0" in kollar),
                  moe=(M.MOE_E if "moe8" in kollar else 0))
    M.init_(model, "gpt", 1.0, a.L, sifir=("sifir" in kollar))
    return model


def main():
    a = ayrıştır()
    kollar = set(a.kol.split("+"))
    assert kollar <= DESTEK, f"bu kodda olmayan kol: {kollar - DESTEK}"
    W, R, cihaz = kur_dagitik(a)
    ana = R == 0
    torch.manual_seed(a.tohum); np.random.seed(a.tohum)
    model = model_kur(a, kollar).to(cihaz)
    N = M.govde_param(model)
    adim = a.adim or int(a.token // a.tok_adim)
    assert adim > 0 and a.tok_adim % (a.mikro * W) == 0, "tok_adim, mikro × süreç sayısına tam bölünmeli"
    nmik = a.tok_adim // (a.mikro * W)
    lr_tepe = 1.5e-3 * 256 / a.d * a.lr_carp  # merdiven.py GPT kuralı (muonw da bunu kullanır)
    M.MUON_LR = a.muon_lr
    optlar = M.opt_kur(model, a.kol, lr_tepe)
    isinma = max(20, int(0.02 * adim))
    sarili = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cihaz.index] if a.cihaz == "cuda" else None) if W > 1 else \
        (torch.compile(model) if DERLE else model)
    dt = np.uint16 if a.dtype == "uint16" else np.uint32
    tr = np.memmap(a.veri, dtype=dt, mode="r"); va = np.memmap(a.dogrulama, dtype=dt, mode="r")
    os.makedirs(a.kayit, exist_ok=True)
    yol = os.path.join(a.kayit, "son.pt")
    s0, pos = 0, 0
    if a.devam and os.path.exists(yol):
        k = torch.load(yol, map_location=cihaz, weights_only=False)
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
                 args=vars(a), zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
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
        for o, b0 in optlar:
            for g in o.param_groups:
                g["lr"] = lr_at(s) if b0 == lr_tepe else b0 * lr_at(s) / lr_tepe
        if pos + a.tok_adim + 1 > len(tr):
            pos = 0
            if ana:
                print("UYARI: veri bitti, baştan alınıyor", flush=True)
        top = torch.zeros((), device=cihaz)
        for i in range(nmik):
            off = pos + (i * W + R) * a.mikro
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
        if os.path.exists(os.path.join(a.kayit, "DUR")):  # güvenli duraklatma: adım bitince kaydet ve çık
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
            sonuc = dict(kol=a.kol, L=a.L, d=a.d, N=N, adim=adim, token=adim * a.tok_adim, val_nat=round(v, 5),
                         flop=hesap_duzelt(kollar, N, a, adim), sure_sn=round(time.time() - t0, 1),
                         surec=W, zaman=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(sonuc, open(os.path.join(a.kayit, "sonuc.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("SONUC", json.dumps(sonuc), flush=True)
    if W > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
