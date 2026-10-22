# F5 sayı doğrulaması: ham dosyalardan yeniden hesap vs kuyruk raporları

Üretim: `scripts/make_report.py` (2026-10-18 03:24). Kuyruk kodu kullanılmadı; formül bağımsız yeniden yazıldı: m = (C_taban/C_kol)·(L_taban/L_kol)^(1/b).

**Sonuç: 137 karşılaştırmanın 137'i tutuyor, 0 uyuşmazlık.**

## Notlar ve gözlemler (rapor/sayaç tutarlılığı)

- sonuc.json val_nat ile gunluk.jsonl'deki son doğrulama (aynı koşuda, 0–3 adım önce) arasındaki en büyük fark 0.00060 nat (gürültü düzeyi; tutarlı).
- FLOP yedek formülü (6·(N+V·d)·token) kayıtlı flop alanıyla sade kollarda 0.0e+00 göreli farkla aynı (n=40); flop alanı olmayan 36 eski satır bu formülle hesaplandı.
- S3_taban: sonuc.json 'sure_sn' = 0.75 sa, ama günlük zaman damgalarından toplam duvar saati 16.62 sa (duraklat-devamda sayaç yalnız son parçayı sayıyor; egit_h.py sure_sn'i kayıtlamıyor). Süre karşılaştırması için sure_sn KULLANILMAZ.
- S3_kanit: sonuc.json 'sure_sn' = 12.05 sa, ama günlük zaman damgalarından toplam duvar saati 14.15 sa (duraklat-devamda sayaç yalnız son parçayı sayıyor; egit_h.py sure_sn'i kayıtlamıyor). Süre karşılaştırması için sure_sn KULLANILMAZ.
- KANIT_ON_KAYIT.md başlığındaki 'ölçülen hız ≈627.384 token/sn ⇒ ≈0,8 saat' S3_taban'ın yalnız son parçasından (2708 sn) türetilmiş: gerçek hız ≈28387 token/sn (1,699 milyar token / 16.6 sa). Ön kayıt dosyası değişmez (sha256); bu bir yanlış türetilmiş yardımcı sayıdır. Öngörüler (bpb bandı) FLOP'tan türediği için etkilenmez.
- Eğim kuralı raporlarda üç farklı: F2 raporu b = 0,0932 (küçük merdiven), F3 raporu b = 0,0609 (S1–S2 taban), F4/otomatik rapor b = 0,0595 (S1–S3 taban). Bu yüzden aynı koşunun m'si raporlar arasında değişiyor (ör. S1 paket ayarlı: 1,538 / 1,963). F5'te tek ana kural ve duyarlılık bandı verilir (bkz. paper).
- Ayarlı tabanın kendi S1–S3 eğimi b = 0.0548 (varsayılan tabanınki 0.0595). Ayarlı taban için daha tutarlı yerel eğim budur; K4' (≥ 2,0) bu eğimle 2.32 çıkıyor.
- m ≥ 2,0 eşiğini 85M'de aşmak için ayarlı taban eğiminin b ≤ 0.0735 olması gerekiyor.
- S3 koşularının üç günlüğünde (S3_taban, S3_kanit, S3_taban_lr2.8) aynı adım 10.619'da eğitim kaybı ≈0,56–0,72 nat'a düşüyor (komşu adımlar ≈2,7–2,9): ortak veri akışındaki tek bir yarı-boş/tekrarlı parti; doğrulama kaybını etkilemez, ama günlük grafiği çiziliyorsa süzülmeli.
- Küçük merdivende paket kolları derlemesiz (eager), varsayılan taban (8 Ekim) derlemeli koşuldu; derlemesiz–derlemeli taban farkı +0,0059 nat (g3, tek ölçüm) ⇒ küçük merdiven m'leri hafif muhafazakâr (≈%6).
