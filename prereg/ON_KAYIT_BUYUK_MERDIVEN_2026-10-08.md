# Ön kayıt — büyük merdiven ve Hyperball (A16) (8 Ekim 2026 ~22:15, hiçbir koşu yapılmadan)

**Neden:** Deep Research 2 (kaynaklar\DEEP_RESEARCH_2_..._OZET.md): 0,4M–7,4M'de ölçülen m'nin büyükte korunacağına kanıt yok; güvenilir vekil için ≥3 basamak, her basamakta aynı kural, tohum gürültüsü ve m'nin eğimi. D10 v2'nin en küçük boyda bozulması da aynı yönde. Aranan şey: **kazancı model büyüdükçe artan** bir yöntem (devrim adayı tanımı, DURMA_KURALI).

**Ne zaman:** kanıt koşusu bittikten sonra (cumartesi 10 Ekim 08:00 sınırı), Mert BTK'dan dönünce (10–11 Ekim laptop eğitimde). Kanıt koşusunun dosyalarına (egit_evren.py, merdiven.py, kanit_kos.py) dokunulmaz; yeni kol ayrı dosyada: `EVREN_TEK_ATIS\egit_h.py`.

## Basamaklar (gövde parametresi, egit_evren / merdiven GPT, T=512, vocab 8192)
| Basamak | L × d | Gövde (taban / paket) | Token (20/param) |
|---|---|---|---|
| S1 | 8 × 512 | 25,17M / 25,21M | ≈0,50B |
| S2 | 10 × 640 | 49,17M / 49,22M | ≈0,98B |
| S3 | 12 × 768 | 84,95M / 85,03M | ≈1,70B |
Veri `kanit_train.bin` (≈4B token, tekrar yok), doğrulama `val.bin`; LR kuralı egit_evren (lr_tepe = 1,5e-3·256/d); tok_adim 131072; doğrulama aralığı %5 (aynı boy taban eğrisi için).

## Kollar
- **taban** (AdamW, kosinüs) — her basamakta.
- **paket** = canon+muonw+qknorm+bigram+wsd (bugünkü en iyi tarif) — her basamakta.
- **paket-h** = canon+**muonh**+qknorm+bigram+wsd (Hyperball: gövde 2B ağırlıklarının ve güncellemelerinin Frobenius normu sabit, ağırlık azaltma yok; η = --h_tepe).

## Hyperball öğrenme hızı (yalnız S1, sonuç görülmeden kural)
- η ∈ {0,004 · 0,008 · 0,016} tam S1 koşusu (0,008 ≈ muonw'nin göreli adımı: lr·0,2/σ, HESAP). En düşük son val kaybı seçilir; S2 ve S3'te aynı η. Kenarda (0,004 ya da 0,016) çıkarsa bir adım daha (0,002 ya da 0,032) eklenir, bu da önceden yazılı.

## Ölçü ve kararlar
- m: katsayi.py'nin Ek 3a yöntemi (aynı boy taban eğrisi; kayıp → hesap). Ayrıca oran **r = m(paket-h) / m(paket)** her basamakta.
- **Tohum gürültüsü:** S1'de paket ve paket-h ikinci tohumla tekrarlanır; |Δm|/m bandı gürültü sayılır (beklenti ≈±%5, merdivendeki gibi).
- **Öldürme (rapordan):** S2 ve S3'te r < 1,03 → A16 öldü.
- **"Boyla büyüyor" işareti:** r(S1) < r(S2) < r(S3) ve r(S3) − r(S1) > gürültü bandı. Ayrıca paket için m(S1), m(S2), m(S3) eğimi raporlanır (küçük merdivendeki "boyla küçülüyor" bulgusunun büyük basamaklarda tekrarı ya da çürümesi).
- Sonuç ne çıkarsa çıksın yazılır; post-hoc değişiklik "post-hoc" diye etiketlenir.

## Sıra ve tahmini süre (hız kanıt koşusunun hız ölçümünden doldurulacak; tahmin)
S1: taban, paket, paket-h ×3 η, paket ve paket-h tohum 2 (7 koşu) → S2: taban, paket, paket-h (3) → S3: taban, paket, paket-h (3). Toplam 13 koşu.

## A17 (model büyütme / Untied-Grow; Chen ve ark. 2026) — kod hazır (~22:20), koşu yok
- Kol **paket-b** = canon+muonw+qknorm+bigram+wsd+**buyuyen** (`egit_h.py`): yarım derinlikle başla, adımların yarısında her blok ardışık kopyalanır, optimizer durumları kopyalara geçer; hesap yarım gövdeyle sayılır (N_yarim). CPU sınamaları geçti: büyüme, büyümeden sonra kayıt/devam, Hyperball ile birlikte.
- Mimariye özel LR (Chen): S1'de --lr_carp {0,5 · 1 · 2}; en düşük son val kaybı seçilir, S2–S3'te aynı. Ölçü r_b = m(paket-b) / m(paket); öldürme r_b < 1,05 (S2 ve S3); boyla büyüme işareti r_b(S1) < r_b(S2) < r_b(S3) ve fark > gürültü.
- Ek koşu: S1'de 3 (LR) + S2 1 + S3 1 = 5 koşu (toplam 18).

## Kuyruk betiği (8 Ekim ~23:20, koşu yok)

`EVREN_TEK_ATIS\buyuk_merdiven.py` yukarıdaki sırayı ve seçim kurallarını uygular (S1'de en iyi η ve lr_carp en düşük son val_nat; kenar uzatması 0,002 / 0,032; S1 ikinci tohum; S2–S3 taban, paket, paket-h, paket-b). Mikro boy her basamakta hız testiyle seçilir. CPU'da minik modelle sınandı; başlatılmadı (Mert 8 Ekim 23:20: “yenisini başlatma, sakın çok uzun saatler olmasın”). Tahmini süre 6–9 gün; başlatma Mert'in açık “başla”sıyla.

- 12 Ekim 2026 (F2'den ÖNCE): tabana da S1 LR denemesi ve ayarlı taban kuralları eklendi → `ON_KAYIT_MERDIVEN_2026-10-08.md` Ek 7c (sha256 89cd537af46f17f8e6f0332a89b740ff2fa507104d6549d1c6ea909576b6f3ae).
