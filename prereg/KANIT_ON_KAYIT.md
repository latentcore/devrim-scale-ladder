# Laptop kanıt koşusu — ÖN KAYIT (otomatik, koşudan ÖNCE yazıldı: 2026-10-16 17:11:19)

Bu dosya ana koşu başlamadan yazılır ve sonra DEĞİŞTİRİLMEZ (sha256 kanit_durum.txt'de). Yöntem: `KANIT_PLAN.md`.

- Tarif: `canon+muonh+qknorm+bigram+wsd+buyuyen` · boy L=12, d=768 (gövde 85.0M parametre) · 1.70 milyar token (20 token/parametre, merdivenle aynı) · adım 12962 × 131072 token · T=512 · lr_carp 2.8 · mikro 4096
- Ölçülen hız ≈627,384 token/sn ⇒ tahmini süre ≈0.8 saat · hesap C = 9.31e+17 işlem (merdivenin en büyük noktasının 129 katı)
- Ölçüm: merdivenle aynı val bpb (val.bin, T=512, 2M token, özel tokenlar hariç); ayrıca HellaSwag, ARC-E/C, MMLU, TurkishMMLU (olc.py).

## Öngörüler (merdiven g1–g4 noktalarından)

- P1 saf kuvvet yasası (g2–g4, eğim 0.0792): **0.829 bpb**
- P2 doyan yasa (g1–g4): **1.113 bpb** (E = 1.096)
- P3 taban doyan yasası × tarif çarpanı (m = 3.06): **0.804 bpb**
- **Ön kayıtlı nokta öngörü: 0.915 bpb, bant ±0.175** (P'lerin ortalaması; bant = yayılım/2 + 0,02, en az 0,03)
- Sağduyu çıpası: SmolLM2-135M bizim val metninde ≈0,808 bpb (8 Ekim), eğitim hesabı ≈1,6e21 (model kartı, İDDİA) — bizim hesabın ≈1719 katı.

## Karar kuralı (önceden yazılı)

- Ölçülen bpb bant içindeyse (0.741–1.090): merdiven yasası ≈129 kat dış taşımada TUTTU ⇒ büyük hesap öngörüleri (Haiku'ya %) güvenilir sayılır.
- Bandın üstündeyse (> 1.090): yasa büyükte iyimser ⇒ büyük ölçek öngörüleri bu oranda kötüleştirilir; kazancın büyükte kaybolup kaybolmadığı ayrıca sınanır.
- Bandın altındaysa (< 0.741): yasa kötümser ⇒ büyükte beklenenden iyi (en ilginç sonuç; tekrar ister).
- Kıyas (ön kayıtlı değil, bilgi): aynı testlerde SmolLM2-135M/360M/1.7B (kalibrasyon.json).
