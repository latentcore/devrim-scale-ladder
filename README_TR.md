# Devrim: bilinen eğitim hilelerinin paketi küçük dil modellerinde ne kadar hesap kazandırıyor?

Tek bir laptop GPU'sunda (RTX 4060, 8 GB), 0,4M–85M parametre arasında, ön kayıtlı bir boy merdiveni. **Taslak v0, 18 Ekim 2026; hakem değerlendirmesinden geçmedi.**

**Soru:** Learning rate'i ayarlanmış bir AdamW tabanına göre, yeni hilelerin bir paketi (canon katmanları, Muon türü optimizer'lar, QK-norm, hash'li bigram gömme, WSD takvimi, derinlik büyütme) hesap açısından ne kadar kazandırıyor ve model büyüdükçe bu kazanç değişiyor mu?

**Bulgular (ayrıntı ve sınırlar `paper/main.pdf` içinde):**
- Hesap çarpanı *m*, tabanın kendi yerel eğimiyle (b = 0,0548) 25M–85M'de ≈2,3–2,7; küçük merdivenin daha dik eğimiyle (b = 0,0932) ≈1,8–2,0. *m* eğime bağlı: ön kayıtlı "m ≥ 2" kuralı 85M'de yalnız b ≤ 0,0735 için tutuyor.
- Eğimden bağımsız ölçü olan göreli kayıp azalması model büyüdükçe **büyümüyor**: ≈%10 (1,3M) → %5 (7,4M) → %4,1 (25M) → %3,9 (49M) → %3,1 (85M).
- Yalnız tabanın learning rate'ini ayarlamak ≈1,25–1,5 kat hesap değerinde.
- 49M ve 85M tek koşu; taban LR'si 49M ve 85M'de yeniden ayarlanmadı; paketin (Recipe A) LR'si hiç ayarlanmadı.

**Bunu göstermiyor:** kazancın 85M'nin üstünde sürdüğünü, büyüdüğünü ya da kaybolduğunu; tek tek bileşenlerin gerekli olduğunu; tarifin alt görev başarısını artırdığını.

**Nasıl doğrulanır (GPU gerekmez):** `python scripts/make_report.py` bütün sayıları, tabloları ve şekli `results/` içindeki ham dosyalardan yeniden üretir ("137 / 137 tamam"); `python prereg/verify_hashes.py` ön kayıt hash'lerini bugünkü dosyalarla karşılaştırır. Eğitim koşularının kendisi tek komutla tekrarlanamaz (her 85M koşusu 14–17 saat).

**Not:** Kodun ve altyapının çoğu yapay zekâ ajanlarıyla (Claude) yazıldı; hedefi, kısıtları ve hangi deneyin koşulacağını yazar belirledi, sonuçları yazar gözden geçirdi. Ön kayıt belgeleri Türkçe orijinal hâlindedir (değiştirilmedi).
