# Devrim — "Ne zaman bırakırız" kuralı (proje çapında ön kayıt)

Yazıldı: 8 Ekim 2026 ~02:00, Mert'in "başla"sıyla, ikinci tohum koşusundan ÖNCE. Yazan: Claude (Opus 5.5). Değişiklik yalnız Mert'in açık kararıyla; her değişiklik tarihiyle bu dosyanın sonuna eklenir, eski metin silinmez.

**Neden var:** MertFormer, AKSİYOM ve MertCore'da "ne görürsek bırakırız" önceden yazılmamıştı; işler uzadı, sonuç tartışmalı kaldı. Devrim'de her aşamanın GEÇME ve ÖLDÜRME ölçütü sonuçtan önce burada durur. Sonucu görüp ölçüt değiştirmek yok.

## Aşamalar

**Aşama 1 — J hakeminin kalibrasyonu: BİTTİ, TUTTU** (8 Ekim 00:24; devir notu "7j sonuç").

**Aşama 2 — tohum tekrarı (tohum 1; taban + düz LR çifti; BLiMP dilbilgisi sondası açık).**
- GEÇER: taban J < 0 VE düz J > 0 VE düz r_early ≥ 0,4 VE düz final_surp ≤ taban final_surp + 0,2.
- Biri tutar biri tutmazsa: tohum 2 koşulur (tek ek hak). Üç tohumun ikisinde tutmazsa J yöntemi ELENİR: Devrim kapanmaz, ama aday seçiminde J kullanılmaz, yalnız Aşama 4 ölçütü kalır.
- Dilbilgisi sorusu (ayrı kayıt): Frank CHILDES'te "düz LR dilbilgisini değiştirmiyor" buldu. Beklentimiz aynı: düz koldaki BLiMP son doğruluğu tabandan en çok ±0,02 farklı. Fark bundan büyükse yeni bulgu sayılır ve tohum 2'yle tekrar edilir.

**Aşama 3 — aday elemesi.**
- Bütçe: en çok 12 aday YA DA 3 hafta (29 Ekim 2026), hangisi önce dolarsa.
- Frank'in denediği aileler TEKRAR EDİLMEZ: LR büyüklüğü/şeması, init, weight decay, batch büyüklüğü, tekrar, grokfast.
- Her aday için koşudan önce bir ön kayıt satırı: hipotez + geçme eşiği + öldürme ölçütü.
- GEÇER (iki tohumda da): aday, aynı bütçede taban kadar iyi öğrenir (doğrulama kaybı tabandan kötü değil; tolerans aynı tohum çiftlerinin farkından ölçülür, yoksa 0,005 bpb) VE en az biri: J ≥ taban J + 0,5, ya da BLiMP son doğruluğu ≥ taban + 0,02, ya da küçük merdivende (Aşama 4 aleti) eğim üstünlüğü.
- ÖLDÜRME: bütçe dolar ve hiçbir aday geçmezse Aşama 3 biter; "4060'ta ölçütü geçen müdahale bulunamadı" raporu yazılır.

**Aşama 4 — ölçekleme üssü (Δα), devrim ölçütünün kendisi.**
- Önce alet kalibrasyonu (D10, mini-Gundlach): aynı merdivende Transformer ile LSTM karşılaştırılır. Bilinen sonuç: Transformer'ın eğimi daha dik. GEÇER: Δα = α_T − α_LSTM > 0 VE en büyük boyutta Transformer kaybı LSTM'den düşük. Tutmazsa alet bozuk sayılır, Aşama 4 kararı verilmez, önce alet düzeltilir.
- Aday testi: aday ve taban, aynı tarifle en az 3 (tercihen 4) model boyutunda; her boyutta parametre başına token sabit; en az 2 tohum.
- GEÇER: Δα = α_aday − α_taban > 0, tohum ve boyut yeniden örneklemeli %90 aralığı 0'ı içermiyor, VE en büyük boyutta aday kaybı tabandan düşük.
- ÖLDÜRME: aralık 0'ı içerir ya da Δα ≤ 0 ise aday ölür; Aşama 3 bütçesi kaldıysa oraya dönülür.

**Aşama 5 — açık yayın ve bağımsız tekrar.**
- Kod, veri sürümü ve bütün ön kayıtlar açılır; ≥10 kat büyük ölçekte en az bir dış tekrar istenir.
- GEÇER: dış tekrar aynı yönü bulur. Devrim iddiası ancak bundan sonra kurulur.

## Genel tavan

- Aşama 2–4 için en çok 8 hafta: **3 Aralık 2026**. O tarihte Aşama 4'ü geçen aday yoksa Devrim KAPANIR (silinmez; bütün sonuçlar negatif rapor olarak yazılır, portfolyo kanıtı olur).
- Uzatma yalnız Mert'in açık kararıyla, bir kez, en çok 4 hafta.

## Davranış kuralları

- Gece 02:00'den sonra "buldum" ilanı yok; iyi görünen sonuç önce ikinci tohumla tekrar edilir.
- Tek koşuya inanılmaz; tek tohum sonucu "işaret"tir, kanıt değil.
- Her koşunun çıktısı, ön kaydı ve hükmü devir notuna yazılır (`00_DEVIR_TESLIM.md`).
- Yeni ana hat açılmaz (8 Ekim budaması); Devrim içinde de aday listesi dışına çıkılmaz.

## Değişiklik kaydı

- 12 Ekim 2026 ~12:56 — Mert'in açık kararıyla (plan sorusuna cevap: "Evet, 24'e çıkar") Aşama 3 aday bütçesi 12 → 24; tarih sınırı (29 Ekim 2026) aynı. Gerekçe: yol haritası F1A (9 kol) + F1B döngü kolu = 10 yeni aday; 11 kullanılmıştı. Tarif bayrakları ve ölçüm/kontrol işleri (T1, T2, F1K) bütçeye sayılmaz. Ön kayıt: `ON_KAYIT_MERDIVEN_2026-10-08.md` Ek 7.
