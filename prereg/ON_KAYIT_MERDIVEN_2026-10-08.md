# Ön kayıt — Merdiven deneyleri (8 Ekim 2026 gecesi)

Yazıldı: 8 Ekim 2026 ~02:20, koşulardan ÖNCE (kuyruk ana koşu bitince ~05:00'te başlar). Yazan: Claude. Bağlı kural: `DURMA_KURALI.md` (Aşama 3–4).

## Alet

- `d1_frank\merdiven\merdiven.py`: aynı ClimbMix verisi (8192 BPE, sabit sıra), adım başına 32.768 token, bağlam 512, AdamW (0,9/0,95, wd 0,1), ısınma %2 + kosinüs (tepe LR'nin %10'una), parametre başına 20 token (gövde parametresi = gömme ve çıkış başı hariç).
- Boyutlar (gövde parametresi): GPT g1 0,39M · g2 1,33M · g3 3,15M · g4 7,4M · g5 14,2M; LSTM l1 0,41M · l2 1,33M · l3 3,2M · l4 6,6M.
- Ölçü: 2M token doğrulama kaybı (nat/token) ve bpb. Eğim: log L = c − α log N (log-log); ayrıca ortak-E uyumu (`egim.py`).
- LR: g1,g2 ve l1,l2'de {0,5; 1; 2} çarpanları denenir, iki boyutun ortalamasında en iyisi o mimarinin bütün koşularına sabitlenir. Adaylar tabanla AYNI LR'yi kullanır.

## D10 — alet kalibrasyonu (önce bu)

- Bilinen: Transformer'ın ölçekleme eğimi LSTM'den diktir (Gundlach 2025 ve öncesi).
- GEÇER: α_GPT − α_LSTM > 0 (log-log) VE en büyük boyutta GPT kaybı < LSTM kaybı.
- Tutmazsa: alet bu ayarda bozuk sayılır; aşağıdaki adaylar için Δα hükmü VERİLMEZ (yalnız ham sayılar yazılır).

## Adaylar (GPT, g1–g4, tohum 1; sonra en iyisi tohum 2 ve g5)

| Aday | Hipotez (neden çocuk benzeri hızlanan getiriye bağlı) | Beklenti |
|---|---|---|
| `duz_dusuk` | Frank'in J kaldıracı (sabit LR ×0,1 + init ×0,3): J'deki hızlanma ölçekleme üssünü de değiştiriyor mu? | Δα ≈ 0 ya da negatif, kayıp kötü (J yapay olabilir) |
| `seq_mufredat` | Kısa bağlamdan uzuna (64→128→256→512, dört eşit parça): çocuk önce kısa söz duyar | belirsiz |
| `mtp2` | t+2 tahmini için ek baş (kayıp 1 + 0,5): yayınlarda fayda boyutla artıyor | Δα > 0 olabilir, küçük boyutta kayıp kötü |
| `veri_mufredat` | Aynı belge kümesi kolaydan zora (unigram sürprizi, kısa belge büzülmeli): Frank'in denenmemiş "müfredat" fikri | belirsiz |
| `zpd` | Yakınsal gelişim: p≈0,5 tokenlara ağırlık (4p(1−p), taban 0,1); bilinen bilginin üstüne inşa | belirsiz (sıra dışı) |

- Gürültü: taban tohum 2 ile ε_α = |Δα(taban t1, taban t2)| ve ε_L = |L_t1(g4) − L_t2(g4)|.
- TEK TOHUM İŞARETİ: Δα > max(2ε_α; 0,005) VE L_aday(g4) ≤ L_taban(g4) + ε_L. İşaret, kanıt değildir.
- AŞAMA 4 GEÇİŞİ (DURMA_KURALI): ≥2 tohum, ≥3 boyut, Δα aralığı 0'ı içermiyor, en büyük boyutta aday kaybı tabandan düşük. Bu gece en çok bir aday için iki tohum koşulur.
- ÖLDÜRME: Δα ≤ max(2ε_α; 0,005) ya da en büyük boyutta kayıp tabandan kötü (ε_L'den fazla) → aday ölür, yeniden denenmez (Aşama 3 bütçesinden düşer: 12 adayın 5'i bu gece).

## Sınırlar (önceden yazılı)

- Boyutlar küçük (0,4M–14M gövde). Burada çıkan pozitif sonuç "işaret"tir; devrim iddiası için Aşama 5 (≥10 kat büyükte dış tekrar) şart.
- Her boyut kendi tam LR şemasıyla ayrı eğitilir (aynı modelin ara noktaları kullanılmaz).
- Sıra sabit, tohum yalnız başlatmayı değiştirir.

## Ek (02:20, koşulardan önce) — 6. aday

| `veri_budama` | Sorscher ve ark. 2022 ("Beyond neural scaling laws"): iyi bir budama ölçütüyle veri seçmek kuvvet yasasını aşan ölçekleme verebilir. Her boyutta 2 kat havuzdan (doğal sıradaki ilk 2D token) zorluk puanının orta %50'si tutulur (en kolay ve en zor çeyrekler atılır), belgeler doğal sırada kalır; budama oranı her boyutta sabit | Δα > 0 (yazının öngörüsü); kayıp belirsiz |

- Aynı geçme/öldürme ölçütleri geçerli. Bu gece aday sayısı 6 (12'lik bütçeden).

## Ek (05:10, D10 koşulmadan önce) — LSTM alet düzeltmesi

- LR taramasında LSTM l1/l2 kaybı 7,33–7,36'da kaldı = unigram entropisi (7,355): LSTM yalnız kelime sıklığını öğreniyordu (bilinen LSTM düzlüğü; bf16'da küçük gradyanlar kayboluyor). GPT etkilenmedi.
- Düzeltme (yalnız LSTM): tam hassasiyet (autocast kapalı) + unutma kapısı sapması 1 (Jozefowicz 2015). Sınama: l1, 250 adım → 6,91 (düzlükten çıkıyor), 149k token/sn.
- LSTM taraması yeniden koşulur; GPT tarama sonuçları geçerli (GPT kodu değişmedi). D10 ölçütleri DEĞİŞMEDİ.
- Uyarı (önceden yazılı): küçük LSTM'ler düzlükte kalırsa LSTM eğimi yapay olarak dikleşir ve D10 yanlış yönde çıkabilir; bu durumda D10 "alet bozuk" sayılır, aday hükmü verilmez.

## Ek (05:27, D10 sonuçları görülmeden; LSTM D10 koşusu henüz başlamadı) — D10 aralığı

- LR taraması bitti: GPT çarpanı 1,0 (g1+g2 ortalaması 5,249; 2,0: 5,255; 0,5: 5,430), LSTM çarpanı 1,0 (6,778). LSTM l1 ÜÇ LR'de de 7,36'da (unigram düzlüğü, 251 adım yetmiyor); l2 düzlükten çıkıyor (6,19).
- Bu yüzden D10 hükmü iki mimarinin AYNI ve düzlük-dışı aralığında verilir: GPT g2–g4 (1,33M–7,4M) ile LSTM l2–l4 (1,33M–6,6M), log-log α. Ölçütler aynı (Δα > 0 VE en büyük boyutta GPT kaybı < LSTM). l1/g1 dahil hesap da şeffaflık için yazılır ama hükme girmez.
- Adayların değerlendirmesi değişmedi (GPT g1–g4; GPT'de düzlük yok: g1 5,69).

## Sonuç (11:50, koşulardan sonra)

- D10 GEÇMEDİ (Δα −0,0036); aday Δα hükmü verilmedi. Kayıp koşuluyla 4 aday öldü, 2'si (mtp2, veri_budama) bekliyor. Ayrıntı: `00_DEVIR_TESLIM.md` "7m sonuç", `d1_frank\merdiven\gece_rapor.md`.

## Ek 2 (8 Ekim ~14:30, D10 v2 koşulmadan önce) — alet düzeltmesi 2 ve ikinci tur adaylar

**Teşhis.** Gece LSTM'leri eğitimin büyük kısmında unigram düzlüğünde (~7,4) kaldı: l2 80. adımda 7,49; l3 196. adımda 7,43; l4 200. adımda 7,30. Küçük LSTM'ler daha uzun süre düzlükte kaldığı için LSTM eğimi yapay olarak dikleşti; 05:10 ekindeki uyarı tam olarak bunu öngörmüştü. Neden: gömme 0,02 ölçekle başlıyor; normsuz LSTM'e minik girdi gidiyor, kapılar yalnız sapmayla çalışıyor.

**Düzeltme (`MERDIVEN_LSTM=v2`, yalnız LSTM):** gömme çıkışına ve LSTM çıkışına RMSNorm. Tek tanı koşusu (l1, LR çarpanı 1): v1 7,368 (bütün koşu düzlükte) → v2 5,977. GPT kodu değişmedi; GPT sonuçları geçerli. v2 sonuçları ayrı dosyalara yazılır (`lr_tarama_v2.jsonl`, `merdiven_sonuc_v2.jsonl`); eski LSTM satırları karışmaz.

**LR:** aynı yöntem. l1 ve l2'de {0,5; 1; 2} denenir, iki boyutun ortalamasında en iyi çarpan LSTM'in bütün koşularına sabitlenir. Tanı koşusu, l1 × 1 satırı olarak sayılır.

**D10 v2 GEÇER ölçütü (ikisi birden tutmalı, değilse GEÇMEDİ):**
1. Asıl aralık (ilk ön kayıttaki tasarım; 05:27 daraltmasının gerekçesi olan düzlük artık yok): GPT g1–g4 ile LSTM l1–l4 arasında log-log Δα = α_GPT − α_LSTM > 0.
2. Gece aralığı (g2–g4 ile l2–l4): Δα > 0.

Ayrıca en büyük boyutta GPT kaybı LSTM'den düşük olmalı.

**D10 v2 GEÇMEDİ ise (D10 v3):** g5 (14,2M) ve l5 (12,8M) koşulur; aynı ölçütler g1–g5 ile l1–l5'te uygulanır. Aday koşuları bu arada kayıp ölçütüyle sürer.

**Bekleyen iki aday (mtp2, veri_budama):** D10 v2 ya da v3 geçerse eski ölçütle hükümlenir. Eşik Δα > 0,005 VE g4 kaybı ≤ taban + 0,0141. Mevcut sayılar: mtp2 +0,0049, veri_budama +0,0018; ikisi de eşiğin altında. Alet geçerse ikisi de ÖLDÜ yazılır.

**İkinci tur adaylar** (ADAY_LISTESI 7–11; GPT g1–g4, tohum 1, taban LR, ölçütler yukarıdaki gibi):

| Aday | Tanım (koşudan önce sabit) | Beklenti |
|---|---|---|
| `canon` (7) | Her blokta dikkatten önce (A) ve MLP'den önce (C) artık akışa pencere-4 nedensel kanal-başı evrişim, init N(0; 0,02) | Allen-Zhu iddiası: küçük modelde akıl yürütme ve kayıp kazancı; Δα belirsiz |
| `muon` (11) | Gövdedeki 2B ağırlıklar (qkv, o, f1, f2) Muon ile (Nesterov momentum 0,95, Newton–Schulz 5 adım, LR 0,02 × max(1; satır/sütun)^0,5, ağırlık çürümesi yok). Gömme, baş ve normlar AdamW (taban LR). Aynı ısınma + kosinüs şeması iki optimizere çarpan olarak uygulanır | Kayıp kazancı biliniyor (hız rekorları); Δα belirsiz |
| `dongu2` (10) | Aynı L blok iki kez sırayla uygulanır (ağırlık paylaşımı). Parametre ve token aynı, hesap 2 kat | Her boyutta kayıp kazancı beklenir; Δα sıra dışı soru |
| `ngram` (9) | Eğitimin ilk %25'inde hedef = (1−λ)·tek-sıcak + λ·bigram(x_t), λ 0,5'ten 0'a doğrusal iner. Bigram yalnız o boyutun göreceği token akışından sayılır, düzleştirme 0,01/V. Ölçü düz CE | Erken hızlanma; Δα belirsiz (D6 fikri) |
| `buyuyen` (8) | Başlangıç derinliği ⌈L/2⌉. Adımların yarısında her blok ardışık kopyalanır; tek L'de fazla blok sondan atılır. AdamW moment durumları da kopyalanır. N = son (tam) model | Hesap tasarrufu biliniyor; Δα belirsiz |

- **Kural:** aday hükmü ancak D10 v2 ya da v3 GEÇERSE verilir. Alet geçmezse yalnız kayıp ölçütü uygulanır: en büyük boyutta kayıp tabandan ε_L'den fazla kötüyse aday ÖLÜR.
- **TEK TOHUM İŞARETİ veren en iyi aday:** tohum 2'ye girer. İki tohumda da Δα > 0 ve g4 kaybı tabandan düşükse taban ile aday g5'te koşulur.
- **Aşama 3 bütçesi:** bu tur 5 aday. Toplam 11/12 olur; geriye 12 (kendi kendine müfredat) kalır.

## Ek 2 durumu (8 Ekim ~14:30)

- LSTM v2 LR taraması: l1 × 0,5 6,009 · × 1 5,977 · × 2 6,293. l2 taraması Mert'in "şu an koşu yapma" isteğiyle (14:24) durduruldu; D10 v2 ve ikinci tur adaylar KOŞULMADI. Ölçütler yukarıdaki gibi geçerli; devam `d1_frank\merdiven\kuyruk3.py` ile (bitmiş koşuları atlar).
- `muon` adayı yalnız Muon'dur; ADAY_LISTESI 11'deki μP kısmı uygulanmadı (Muon'un LR'si genişlikten büyük ölçüde bağımsızdır).

## Ek 3 (8 Ekim ~16:21, ilk koşudan ÖNCE) — "Haiku 4.5'e uzaklık" ölçüsü ve hızlı tur

**Neden:** Mert: "laptop'ta Haiku 4.5 seviyesini başarmalıyız; saatlerce koşu bizi %5 bile yaklaştırmıyorsa anlamı yok". Eğim (Δα) ölçüsü Haiku'ya mesafeyi söylemiyor; bu turun birincil ölçüsü **etkin hesap çarpanı m**.

**Ölçü:**
- Hesap: C = 6 · (gövde + çıkış başı) · token (`merdiven.py` `flop_hesap`; dongu2 gövdeyi iki kez sayar, mtp2 ikinci başı, buyuyen ilk yarıda yarım gövde; gömme okuması sayılmaz).
- Taban eğrisi: taban tohum 1 ve 2'nin g2–g4 noktalarından (boyut başına ortalama) `ln L = a − b · ln C`.
- Bir adayın bir boyuttaki çarpanı: m = C_taban(L_aday) / C_aday (tabanın aynı kayba ulaşmak için gereken hesabı ÷ adayın harcadığı). Taban eğrisinin dışına 4 kattan fazla taşan tahmin "dış değerleme" diye işaretlenir.
- İkincil: aynı oran duvar saatiyle (tabanın token/sn hızıyla).
- **Hedef sabiti:** Haiku 4.5 sınıfı ≈ 10²⁴ işlem (vekil: Llama 3 8B ≈ 1,2 × 10²⁴; Haiku'nunki açıklanmadı) ÷ laptop haftası ≈ 10¹⁹ ⇒ H = 10⁵. **Yüzde = 100 · log10(m) / 5** (1,8 kat = %5; 10 kat = %20).
- **Eşikler:** g4'te tohumlar arası fark 0,014 nat ≈ m'de ±%5 ⇒ m ≥ 1,10 "gerçek kazanç", m ≥ 1,8 "%5 kuralı". m tabanımıza göredir; öncü labların zaten kullandığı numaralar Haiku'ya gerçek mesafeyi bu kadar kısaltmaz (raporda ayrıca yazılır).
- Sağlama: taban tohum 2'nin m'si 1 ± 0,05 çıkmalı; çıkmazsa ölçü "gürültülü" diye işaretlenir.

**Sıra (`d1_frank\merdiven\kuyruk4.py`; her yeni kol önce 30 adımlık g1 duman sınaması):**
0. Gece koşulan 6 adayın m'si mevcut veriden (GPU yok).
1. muon: g2'de Muon LR {0,01; 0,02; 0,04} (ayrı dosya `muon_lr_tarama.jsonl`), en düşük g2 kaybı veren LR ile g3 → g4.
2. canon, 3. ngram, 4. buyuyen: önce g3; m₃ ≥ 1,10 ise g4.
5. dongu2: yalnız g3 (hesabı ≈1,6 kat yaktığı için m büyük ihtimalle < 1).
6. **Birleştirme:** m₄ ≥ 1,10 olan bütün kollar tek kolda (`+` ile; dongu2 ile buyuyen birlikte konmaz, m'si yüksek olan kalır) g2, g3, g4.
7. En iyi kol (birleşik ya da tek): tohum 2'de g3, g4 + tohum 1'de eksik g1 → boyut başına m; **m'nin boyutla artması** "büyüdükçe açılan fark" işaretidir (tek başına kanıt değil).
8. Kalan sırada LSTM v2 kalibrasyonu (Ek 2'deki D10 v2).

**Hükümler:** m₃ < 1,10 → kol bu ölçüde ÖLDÜ (g4 koşulmaz). Ek 2'deki Δα ölçütleri ayrıca geçerli (D10 geçerse uygulanır). Aşama 3 bütçesi: bu tur yeni aday yok (5 aday Ek 2'de yazılıydı; birleştirme ve Muon LR taraması aday sayılmaz).

**Dürüst beklenti (koşudan önce):** bilinen numaralar büyük ihtimalle 1,3–3 kat getirir = yolun %2–10'u; "Haiku seviyesi" bu turda beklenmez. Bu turda yok: veri kalitesi sınıflandırıcısı ve öğretmen modelden damıtma (indirme izni gerekir; Claude çıktılarıyla model eğitmek Anthropic kurallarına aykırı).

## Ek 3a (8 Ekim ~16:23, Muon ve öteki yeni kolların sonucu GÖRÜLMEDEN) — ölçü düzeltmesi

- Ek 3'teki sağlama tutmadı: taban noktaları düz log-log çizgisine oturmuyor (g3 çizginin ≈0,07 nat üstünde); bu yüzden taban tohum 2'nin m'si boyuta göre 0,85–1,12 çıktı (beklenen 1 ± 0,05).
- Düzeltme: aday AYNI boyuttaki taban ortalamasıyla (tohum 1+2) karşılaştırılır; kayıp farkı taban eğrisinin eğimiyle (b ≈ 0,093; g2–g4) hesaba çevrilir: m = (C_taban,boy / C_aday) · exp((ln L_taban,boy − ln L_aday) / b). Sağlama yeniden: taban tohum 1 ve 2'nin m'si bütün boylarda 0,97–1,03 (tuttu).
- Eşikler, sıra, hedef sabiti ve yüzde formülü aynı. Gece adaylarının bu ölçüyle g4 çarpanları: veri_budama 1,00, seq_mufredat 0,86, veri_mufredat 0,87, mtp2 düşük, zpd 0,22, duz_dusuk 0,12 ⇒ hiçbiri Haiku'ya yaklaştırmıyor (en iyisi %0).

## Ek 4 (8 Ekim ~16:33, bu bayrakların hiçbiri koşulmadan) — speedrun tarif bayrakları (Haiku-% hattı)

- **Neden:** Mert (8 Ekim): "boş bekleme, geliştir, dene". Bilinen en büyük sabit kazanç kaynağı modded-nanogpt speedrun'ı: aynı hedef kayba (FineWeb, 3,28) 8×H100'de 45 dk → ≈1 dk; README'ye göre hedefe 330M'den az tokenla ulaşılıyor (kazancın bir kısmı FP8/çekirdek gibi sistem işidir; bizim hesap ölçümüz yalnız öğrenme verimini sayar).
- **Bayraklar** (`merdiven.py`; varsayılan yol birebir aynı, CPU'da eski kodla bit bit karşılaştırıldı): `qknorm` (rekor 5), `relu2` (rekor 5), `sifir` = sıfır başlangıçlı o ve f2 (rekor 5), `softcap` = 15·tanh(logit/15) (rekor 9/18), `bigram` = 131.071 satırlık (önceki, şimdiki) token hash gömmesi, sıfırdan başlar (rekor 62), `wsd` = sabit LR + son %45'te %10'a doğrusal iniş (rekor 19/26).
- **Kural:** temel = kuyruk4'ün en iyi kolu (m₄ ≥ 1,10); yoksa g3'te m ≥ 1,0 ise muon, değilse taban. Her bayrak temele tek tek eklenir, g3'te koşulur; **artış = m(temel+bayrak) / m(temel) ≥ 1,05** olanlar tutulur. Tutulanların hepsi tek pakette g2, g3, g4 (tohum 1) ve g3, g4 (tohum 2). Paketin m₄'ü kuyruğun en iyisinden yüksekse yeni en iyi odur.
- **Statü:** bunlar eğim (Δα) adayı değil, tarif iyileştirmesidir; Aşama 3 aday bütçesine sayılmaz, Δα hükmüne girmez. Ölçü Ek 3a (m, Haiku'ya %).
- **Dürüst beklenti:** bayraklar 124M modelde ve FineWeb'de ayarlandı; bizim 0,4M–7M modellerimizde kazanç daha küçük ya da ters olabilir. Paket için beklenti 1,1–1,5 kat (yolun %1–4'ü).

## Ek 5 (8 Ekim ~17:05, Muon g4 sonucu GÖRÜLDÜKTEN SONRA — post-hoc, böyle etiketlenir)

- **Gözlem:** muon (LR 0,04, ağırlık çürümesiz) g2'de m 2,55, g3'te 2,67, ama **g4'te 0,52** (kayıp 3,886; taban 3,655). Eğri: 450. adımda tabanın önünde (4,71 / 4,94), 1350–1800. adımda duruyor, sonra geride. En uzun eğitim en büyük boyda.
- **Hipotez (Moonlight, Liu ve ark. 2025, "Muon is Scalable for LLM Training"):** ağırlık çürümesi olmayan Muon'da ağırlıklar uzun eğitimde şişer; çözüm ayrık ağırlık çürümesi ve güncelleme RMS'ini AdamW'ye eşleme.
- **Yeni kol `muonw`:** gövde 2B ağırlıkları Muon, ayrık ağırlık çürümesi 0,1, güncelleme ölçeği 0,2·√max(satır, sütun), LR ve şema AdamW'ninkiyle aynı (`lr_tepe`, boyuta göre zaten ayarlı). Kapısız g2, g3, g4 (tohum 1).
- **Kural değişikliği (post-hoc):** Ek 4'teki speedrun temeli "en iyi kol yoksa g3'te m ≥ 1 ise muon" idi; muon g4'te çöktüğü için temel artık: en iyi kol (m₄ ≥ 1,10) → yoksa m₄ ≥ 1,0 ise muonw → yoksa taban. Birleştirmede muon ile muonw birlikte konmaz, m₄'ü yüksek olan kalır.
- muon'un hükmü (Ek 3a ölçüsüyle): g4'te m 0,52 < 1,10 ⇒ bu ayarda ÖLDÜ; g2–g3 kazancı büyüklükle korunmadı.

## Ek 6 (8 Ekim ~19:25, sonuçlar GÖRÜLDÜKTEN SONRA yazıldı) — Ek 3–5 ara sonuçları

- Ölçü sağlaması (Ek 3a): taban tohumları 0,97–1,03 ⇒ ölçü çalışıyor; gürültü bandı ≈±%5.
- Tek kollar (tohum 1; m g2 / g3 / g4): muon 2,55 / 2,67 / 0,52 (büyükte çöktü ⇒ ÖLDÜ); muonw 1,22 / 2,05 / 1,40 (Ek 5, post-hoc eklendi); canon – / 1,86 / 1,18; ngram g3 0,99 (g3 kapısı < 1,10 ⇒ ÖLDÜ); buyuyen g3 1,07 (ÖLDÜ); dongu2 g3 0,61 (ÖLDÜ).
- Birleştirme (Ek 3 kuralı; muon ile muonw birlikte konmadı, Ek 5): canon+muonw tohum 1 g1 1,83 / g2 1,64 / g3 2,76 / g4 1,47 (+%3,36); tohum 2 g3 2,79 / g4 1,49 (+%3,44) ⇒ tekrar etti.
- Ön kayıttaki beklenti ("bilinen numaralar 1,3–3 kat = yolun %2–10'u") tuttu: en büyük boyda ≈1,48 kat ≈ %3,4. Kazanç boyla artmıyor (g3 2,76 > g4 1,47) ⇒ boyla büyüyen kazanç işareti YOK.
- Koşuyor: Ek 4 hızlandırma bayrakları ve D10 v2; sonuçları bu bölüme devam satırı olarak eklenir.

## Ek 6 devamı (8 Ekim ~21:25, sonuçlar GÖRÜLDÜKTEN SONRA) — hız paketi

- Ek 4 kuralı (≥1,05 kat artış veren bayrak tutulur) uygulandı: qknorm 1,246, wsd 1,252, bigram 1,111 tutuldu; relu2 1,016, sifir 1,015, softcap 1,002 atıldı (temel canon+muonw, g3).
- Paket canon+muonw+qknorm+bigram+wsd: tohum 1 g2 3,411 · g3 4,898 (dış değerleme) · g4 2,207; tohum 2 g3 4,882 · g4 2,185. Haiku'ya % (g4): +6,87 / +6,79.
- Yorum (post-hoc değil, Ek 3'ün ölçüsüyle): en iyi tarif değişti; kazanç boyla küçülme örüntüsü sürüyor (g3 → g4). Kanıt koşusu paketi kullanır (kol_sec).
- D10 v2 hükmü ve kuyruk5 bu ekin altına eklenecek (kuyruk4 bitince).
- **D10 v2 (kuyruk4 22:06):** LSTM v2 LR taraması l1/l2 ortalaması 0,5 → 5,450 · 1,0 → 5,411 · 2,0 → 5,613 ⇒ çarpan 1,0. g1–g4 / l1–l4: α GPT 0,1463 · LSTM 0,1557 · Δα −0,0094 → TUTMUYOR; g2–g4 / l2–l4: α GPT 0,1606 · LSTM 0,1373 · Δα +0,0233 → tutuyor; en büyük boyda GPT 3,662 < LSTM 3,895 nat. Ön kayda göre hüküm **GEÇMEDİ** (iki aralığın ikisi de tutmalıydı). Yorum (post-hoc, böyle etiketlenir): en küçük boy eğimi bozuyor; g2'den sonra beklenen yön çıkıyor. Deep Research 2'nin "0,4–7,4M güvenilir vekil değil" uyarısıyla aynı yönde. Eğim (Δα) hükümleri bu merdivende verilmez; m ölçüsü (aynı boy taban) D10'a bağlı değil. Eğim için merdiven 25M / 50M / 100M'e taşınır (ADAY_LISTESI kuralı).

## Ek 7 (12 Ekim 2026 ~12:56, İLK KOŞUDAN ÖNCE; Mert'in "başla, hiç durma" kararı ve onaylı plan) — kesintisiz kuyruk (kuyruk6): taban ayarı, birleşim taraması, birleşik tarif, akıl yürütme çekirdeği, bilgi kapasitesi, büyük merdiven genişletmesi

Kaynak: `00_YOL_HARITASI_2026-10-10.md` §15 taslağı + 12 Ekim netleştirmeleri. Kuyruk `d1_frank\merdiven\kuyruk6.py` (usta) ve `kuyruk6_faz.py` (fazlar); koşular derlemesiz; ölçü Ek 3a (m; Haiku'ya % = 100·log10(m)/5, H = 10⁵; ikincil sayı H = 3×10⁴ ile: 100·log10(m)/log10(3×10⁴)). Güvenli duraklat/devam (`KUYRUK6_DUR`, adım sonu checkpoint, 10 dk'da bir checkpoint) matematiği değiştirmez; varsayılan yol CPU'da eski kodla bit bit karşılaştırılır. Aday bütçesi: DURMA_KURALI değişiklik kaydı (12 Ekim, 12 → 24; Mert'in açık kararı). Sıra F0 → F1A → BİRLEŞİM → TOHUM2 → F1B → F1A-2 → F1K → F2 → F3 → F4 → F5; kodu hazır olmayan faz beklenirken hazır olan bağımsız faz koşar.

1. **F0 — T1 taban sağlamlaştırma (ölçüm, aday değil).** Taban, tohum 1, derlemesiz; sonuçlar ayrı dosya `t1_taban_ayar.jsonl`. g3 ve g4'te (LR çarpanı, AdamW ağırlık azaltması): (0,7; 0,1), (1,4; 0,1), (1; 0,05), (1; 0,2). Merkez (1; 0,1) = mevcut taban tohum 1/2 ortalaması (derlemeli; derlemesiz farkı ≈0,001–0,006 nat). g4'te en iyi LR 1,4 ise LR 2,0 eklenir; en iyi LR ve en iyi WD ikisi de merkez dışındaysa ikisinin birleşimi eklenir; "en iyi" = g4 doğrulama kaybı. En iyi ayar g2'de de koşulur (eğim için).
   - **K0:** en iyi g4 kaybı ≤ 0,99 × 3,6552 (eski taban g4 ortalaması) ⇒ "ayarlı taban"; bundan sonra bütün m'ler iki tabana göre yazılır (ayarlı tabanın eğimi ayarlı g2–g4 noktalarından). Paketin ayarlı m₄'ü 1,5'in altına düşerse yol haritasındaki "cilalama biter" kuralı güçlenir. Koşul tutmazsa "taban zaten ayarlı" yazılır.
   - **Dolgu (ölçüm, yalnız F1A kodu hazır değilken):** paket g3'te LR {0,7; 1,4}; paket g4 tohum 3 (gürültü bandı).
2. **F1A — birleşim taraması (paket + X).** paket = canon+muonw+qknorm+bigram+wsd. Ölçü aₖ = m(kol, gₖ) / m(paket, gₖ), tohum 1; g3 ve g4 kapı, g2 bilgi amaçlı. Her yeni kol önce CPU duman + g1'de 30 adımlık GPU duman.
   - muonh (A16): muonw yerine Hyperball (`egit_h.py` sınıfı: gövde 2B ağırlık ve güncelleme Frobenius normu sabit, ağırlık azaltma yok). g3'te η {0,004; 0,008; 0,016}; en iyisi kenardaysa 0,002 ya da 0,032 eklenir; en iyi η ile g2, g4. Tarama satırları `f1a_tarama.jsonl`, en iyisi ana sonuca kopyalanır.
   - buyuyen (A17): paket + büyüyen model (yarım derinlik, adımların yarısında blok kopyalama). g3'te LR çarpanı {0,5; 1; 2}; en iyisiyle g2, g4.
   - muonh+buyuyen: iki taramanın en iyi ayarlarıyla g2, g3, g4.
   - overtok (A21): bigram yerine 2-gram (131.071 satır) + 3-gram (262.139 satır) hash'li giriş gömmeleri, sıfırdan başlar.
   - hub4 (Ö4): dizinin başına 4 öğrenilmiş jeton; bütün konumlar görür; kayıp yalnız gerçek jetonlarda; hesap × (512+4)/512.
   - derin (A15): aynı gövde parametresiyle ≈1,5 kat katman: g1 3×104, g2 5×148, g3 6×210, g4 9×256.
   - moe8 (A19): her blokta 8 uzman, jeton başına en iyi 2 (yeniden normalize), uzman genişliği 2d (aktif parametre yoğun FFN'e eşit), yük dengeleme kaybı 0,01 (yalnız eğitim kaybına). m-hesap aktif parametreyle; m-süre ve tepe VRAM ayrıca.
   - Tarif bayrakları (Ek 4 kuralı, aday sayılmaz): ve (her blokta V×d değer gömmesi, sıfırdan, dikkat değerlerine eklenir), x0 (her blok girişi λa·x + λb·x0, λa=1, λb=0 başlar). Yalnız g3; artış ≥ 1,05 olan birleşime girer.
   - **K1:** geçer a₄ ≥ 1,05; **işaret** a₄ ≥ a₃ ve a₄ ≥ 1,05; öl a₄ < 1,0; arası "kazançsız". Bütün koşular tepe VRAM ve toplam parametreyi de yazar. Tablo parametreleri (bigram, overtok, ve) hesap tutmaz, bellek tutar; raporda ayrıca yazılır.
3. **BİRLEŞİM — paket2.** paket2 = paket + K1'i geçen kollar + tutulan bayraklar; muonh geçtiyse muonw çıkar, overtok geçtiyse bigram çıkar, öteki geçenler eklenir. Hiçbiri geçmezse paket2 = paket. g3'te LR çarpanı {0,7; 1; 1,4} (buyuyen varsa merkez onun en iyi çarpanı), en iyisiyle g2, g4 tohum 1 ve g3, g4 tohum 2. **Yeni en iyi tarif:** paket2'nin g4 m ortalaması (iki tohum) > 2,196 + 2 × gürültü (gürültü = paket2 ile paketin tohum farklarından büyüğü). Rapor: m-hesap, m-süre, iki H ile Haiku'ya %, eski ve (K0 tuttuysa) ayarlı tabana göre.
4. **TOHUM2.** İşaretli her kola tohum 2 (g3, g4); iki tohumda da işaret tutarsa "iki tohumlu işaret" ve S1'e öncelikli.
5. **F1B — akıl yürütme çekirdeği.** Görevler Python'la üretilir, cevap kesin: aritmetik zinciri (mod 100; +, −, ×), k-atlama (rastgele permütasyonda k adım), liste programı izi (sırala, ters, tekrarsız, döndür), Dyck (iki parantez türü, dengeli mi). Eğitim k ≤ 4 (Dyck derinlik ≤ 4); sınav: dağılım içi yeni örnekler ve uzun k = 5–8. Kollar: taban (AdamW, GELU), paket, döngü (paket + aynı bloklar eğitimde 2 tur; sınavda 1, 2, 4 tur). Boylar 2×128, 3×192, 4×256; kollar arasında eşit eğitim hesabı (döngü yarı örnek). Sınav kümeleri eğitimden ÖNCE ayrı tohumla üretilir; sha256'ları ve kesin bütçe Ek 7a'ya F1B koşusundan önce yazılır; eğitim üreteci sınav örneklerini reddeder (kesişim 0 doğrulanır).
   - **K1B:** geçer = aday eşit hesapta uzun-k ortalamasında tabanı ≥ 5 puan geçer VE fark büyük boyda küçülmez; test anı işareti = döngüde tur 1 → 4 uzun k'da ≥ 5 puan artış; öl = taban ≥ aday. Self-consistency (8 örnek, çoğunluk) en iyi kolda; taban doğruluğu < %5 ise "uygulanamaz".
6. **F1A-2.** gptbert (A14): mikro partilerin ¼'ü maskeli (%15 maske, ayrı maske jetonu, çift yönlü dikkat, maskelenen jeton bir önceki konumdan tahmin), doğrulama yalnız nedensel. rholoss (Ö1): referans g1 taban, train.bin'in son 40M jetonundan (eğitimle kesişmez) 20 jeton/parametre; her adımda 2 kat aday dizi, (eğitim kaybı − referans kaybı) en yüksek %50 ile güncelleme; hesaba aday ileri geçişleri ve referansın eğitimi ile ileri geçişleri dahil. K1 aynı. T2 (kontrol, aday değil): taban+ep4 ve paket+ep4 (benzersiz verinin ¼'ü × 4 tur), g3, g4.
7. **F1K — bilgi kapasitesi (ölçüm).** Sentetik biyografi (≈50 bin kişi, 6 özellik); tek kalıp × 100 görülme ve 5 ifade + cümle karıştırma × 100; taban ve paket; kişilerin yarısının soru-cevapları eğitimde, soru-cevap doğruluğu öbür yarıda. Öğrenilen bit = Σ max(0, log2|havuz| − CE_bit); bit/parametre (gövde ve toplam). Kesin ayarlar Ek 7b'ye F1K koşusundan önce. **K1K:** iki tohumda > 3 bit/param = olağanüstü işaret (ayrı ön kayıt); ifade çeşitliliği kazancı < 5 puan ⇒ veri yoğunlaştırmaya yatırım azalır.
8. **Büyük merdiven genişletmesi** (`ON_KAYIT_BUYUK_MERDIVEN_2026-10-08.md` ve `KANIT_PLAN.md` ölçütleri aynen). F1 kazananları `egit_evren.py`/`egit_h.py`'ye taşınır (eski yol CPU'da bit bit); taşınamayan S1'e girmez, nedeni yazılır. S1 (25M): taban, paket, paket-h (η {0,004; 0,008; 0,016}), paket-b (LR {0,5; 1; 2}), ≤ 2 F1 kazananı (önce iki tohumlu işaretliler, sonra a₄ en yüksekler), tohum 2 (paket + en iyi kol). K2: r = m(kol)/m(paket) ≥ 1,05 (muonh 1,03) ve > 2 × gürültü. S2 (49M): taban, paket, K2'yi geçen ≤ 2 kol; K3: r(S2) > r(S1) + gürültü ⇒ S3'te kanıt koşusunun kolu o, yoksa paket (ya da BİRLEŞİM'in yeni en iyisi). S3 (85M): taban + kanıt koşusu (`kanit_kos.py`; P1/P2/P3 ön kaydı ve sha256 koşudan önce otomatik); K4: aktarım m_büyük ≥ 2,0 korundu / 1,1–2,0 küçülüyor / < 1,1 kayboldu, öngörü bandı. Devrim işareti: aynı kol için r S1 < S2 < S3 ve farklar gürültüden büyük. F5: `olc.py` + SmolLM2 cetveli + rapor.
9. **Dürüst beklenti (koşudan önce; TAHMİN, kanıt değil):** K0 tutar (taban ≥ %1 iyileşir) ≈%40; F1A'da en az bir kol a₄ ≥ 1,05 ≈%50; a₄ ≥ a₃ işareti ≈%25; birleşik tarif m₄ ≥ 2,6 (Haiku'ya ≥ %8,3) ≈%35; F1B'de uzun k'da ≥ 5 puan üstünlük ≈%30; S1 → S2 boyla büyüme ≈%10; kanıt koşusunda aktarım "korundu" ≈%25; 29 Ekim'e kadar her anlamda genel Haiku 4.5 < %1.

## Ek 7a (12 Ekim 2026 ~13:47, F1B KOŞUSUNDAN ÖNCE) — F1B sınav kümeleri ve kesin bütçe

- **Sınav kümeleri:** `F1B_akil_cekirdek\sinav\<GÖREV>_<ID|OOD>.tsv` (görev × küme × 500 soru; ID tohum 1001 + görev sırası, OOD 2001 + görev sırası; k rastgele, tekrar eden soru atlanır — Dyck derinlik 1'de benzersiz soru az olduğu için ID'de az temsil edilir). sha256 (tam): ARIT_ID f46d4b0d01f6763424c070540678854c75981c78e8e8b05570e96caaff53d936; ARIT_OOD 317aa67cb4f23f9f9f646d6e478450e4b60876032d558648fa05d1207c054d40; KHOP_ID dcdb1caac8ee3e8da10b5e4da552495c5e2077389179a9f568c9712bb90f2344; KHOP_OOD bdaac44d4ee3cb3e9844ae002a10ad53aec5855c6cfb123220bff8548c94b2d8; LISTE_ID 41d61dacbb7994985cebd75d40a0239dfabae5595f2f3d7dee03822bfae8332f; LISTE_OOD 62e59f7c26d0254c77e816214cd5062f0529719a5e85cfec0ccb574c45141db9; DYCK_ID deb6e94a58e843c568186f828191a86a9048078e2a73864274e654126acc3e8e; DYCK_OOD cf73ac745f42157eb2bc415a20c3bfeadadb0758472fc4a3d7171ce34d6d67dd.
- **Eğitim verisi:** tek akış (tohum 7) bütün kollarda ve boylarda aynı; görev ve k (1–4) eşit olasılıkla; sınav sorusuyla aynı olan örnek reddedilir (sayısı sonuç satırında). Satır başına bir örnek, T = 64, sağdan dolgu; kayıp yalnız cevap jetonlarında ve bitiş jetonunda.
- **Bütçe:** 256 satır/adım; taban ve paket 9.000 adım (≈147M jeton konumu), dongu 4.500 adım (eşit hesap: her adımda iki tur). LR kuralı 1,5e-3·256/d; ısınma %2; taban kosinüs (%10'a), paket ve dongu WSD (son %45).
- **Kollar:** taban (AdamW, GELU); paket (canon + muonw + qknorm + karakter çifti tablosu + WSD); dongu (paket + eğitimde aynı bloklar 2 tur; sınavda 1, 2, 4 tur; ana sonuç tur 2). Boylar b1 2×128, b2 3×192, b3 4×256; tohum 1.
- **Ölçü:** görev ve küme başına tam eşleşme; "OOD ort" = 4 görevin uzun-k doğruluğunun ortalaması. Self-consistency: b3 paket, OOD, 8 örnek, sıcaklık 0,7, çoğunluk oyu.
- **K1B (Ek 7 madde 5 aynen):** geçer = (kol − taban) OOD ort farkı en büyük boyda ≥ 5 puan VE en büyük boydaki fark en küçük boydakinden küçük değil; öl = en büyük boyda fark ≤ 0; test anı işareti = dongu tur 1 → 4 OOD ≥ +5 puan. Dar alan uyarısı: bu görevlerde başarı genel akıl yürütme kanıtı sayılmaz (MertCore'da ARİT-1 %100, ARC %9,5 dersi).
- **İsteğe bağlı dış kıyas:** `HAIKU45_AKIL_TESTI_PROMPT.txt` (OOD'den 40 soru; cevap anahtarı ayrı dosyada). Haiku 4.5'e sorulursa yalnız bu dar alanda kıyas yapılır.

## Ek 7b (12 Ekim 2026 ~13:51, F1K KOŞUSUNDAN ÖNCE) — F1K kesin ayarlar

- **Veri:** `F1K_bilgi_kapasite\bilgi_kapasite.py`; 50.000 kişi (uydurma ad + soyad, 400 × 400 havuzdan benzersiz), 6 özellik: doğum tarihi (12 ay × 28 gün × 100 yıl), doğum şehri (200), üniversite (300), bölüm (100), şirket (263), şirket şehri (200) ⇒ ≈53,2 bit/kişi, toplam ≈2,66M bit. Uydurma kelimeler (gerçek dünya bilgisi yok); kelime düzeyi sözlük (1.873).
- **Koşullar:** tek (her özellik için 1. kalıp, sabit sıra) ve cok5 (özellik başına 5 ifade, her görülmede rastgele ifade ve karışık cümle sırası); her biyografi 100 kez görülür (tur tohumu = tur numarası). Kişilerin ilk yarısının 6 soru-cevabı her turda eğitimde; ikinci yarı yalnız biyografiyle (soru-cevap sınavı orada). Aynı jeton bütçesi (≈450M), T = 256, 64 satır/adım.
- **Kollar:** taban (AdamW, kosinüs) ve paket (canon + muonw + qknorm + 131.071 satırlık bigram hash + WSD); tek boy 4×192 (≈1,77M gövde); tohum 1.
- **Ölçü:** öğrenilen bit = Σ max(0, log2|havuz| − CE_bit(değer | 1. kalıp öneki)) (virgül jetonu sayılmaz); bit/parametre gövde, tablo hariç toplam ve toplam olarak; soru-cevap tam eşleşme (açgözlü, ikinci yarı).
- **K1K (Ek 7 madde 7 aynen):** tablo hariç > 3 bit/param ⇒ olağanüstü işaret, tohum 2 ve ayrı ön kayıt; ifade çeşitliliği kazancı (cok5 − tek, soru-cevap ortalaması) < 5 puan ⇒ veri yoğunlaştırmaya yatırım azalır. Beklenti (literatür, TAHMİN): 100 görülmede ≈1 bit/param; tek kalıpta soru-cevap düşük, cok5'te yüksek.

## Ek 7c (12 Ekim 2026 ~15:50 — F2'den ÖNCE; S1/S2/S3'te hiçbir koşu yokken, F1A koşarken): tabana da LR denemesi

**Gerekçe.** F0'da (Ek 7) K0 tuttu: küçük merdivende taban varsayılan LR'de ayarsızmış (en iyi LR×2,0 — denenen aralığın kenarı; g4 kayıp 3,6552 → 3,5943, −%1,67) ve paketin m₄'ü 2,207'den 1,795'e indi (kazancın ≈%19'u zayıf tabandan). S1–S3 aynı LR kuralını kullanıyor (lr_tepe = 1,5e-3·256/d) ve ON_KAYIT_BUYUK_MERDIVEN'de S1'de yalnız paket LR taraması var ({0,5; 1; 2}, paket-b), taban yok. Bu, büyük ölçekteki m'leri ve K4 hükmünü tabanın aleyhine şişirebilir. Bu ek testi zorlaştırır (daha muhafazakâr); ön kayıtlı hiçbir koşu ya da hüküm silinmez, yanına ikincisi eklenir.

**Kurallar (sonuç görülmeden yazıldı):**
1. S1 (F2): taban ×2,0 tam S1 koşusu (`S1_taban_lr2.0`); ×1,0 = `S1_taban` (zaten planlı). ×2,0 daha düşük son val_nat verirse ×2,8 de koşulur (kenar uzatması, η kuralındaki gibi). En düşük son val_nat → lr_t. ×0,5 koşulmaz (F0'da g2–g4'te LR arttıkça taban kaybı tekdüze düştü).
2. c_S = m(paket | varsayılan taban) / m(paket | ayarlı taban), aynı basamak ve aynı eğim b ile.
3. S2 (F3): lr_t ≠ 1 ve c_S1 > 1,05 ise `S2_taban_lr<lr_t>` koşulur.
4. S3 (F4): lr_t ≠ 1 ve max(c_S1, c_S2) > 1,10 ise kanıt koşusundan ve ön kayıtlı K4 yazıldıktan SONRA `S3_taban_lr<lr_t>` koşulur. K4 iki tabana göre yazılır: ön kayıtlı K4 (varsayılan taban) aynen, K4' (ayarlı taban) yanına. Başlık sayısı (Haiku'ya %) ikisinden DÜŞÜK m'yi kullanır.
5. F2, F3, F4, F5 raporlarında bütün m'ler iki tabana göre (ayarlı taban koşulmayan basamakta "–").
6. K2 ve K3 (r = m(kol)/m(paket)) tabandan bağımsızdır, değişmez. Eğim b ön kayıttaki gibi (varsayılan S taban noktaları, yoksa küçük merdiven).
7. Bunlar aday değil taban ayarıdır: DURMA_KURALI aday sayısı değişmez. Ek süre (TAHMİN): S1 +1–2 koşu, S2 +1 koşu (koşul 3), S3 +≈27 sa (koşul 4); F5 ≈19–20 Ekim yerine en geç ≈21–22 Ekim.

**Uygulama.** `d1_frank\merdiven\k6_buyuk.py` (taban_lr_sec, c_duzeltme, taban_ad; F2/F3) ve `k6_kanit.py` (F4 Ek 7c bloğu, F5 iki taban sütunu; yedek `k6_kanit.py.bak_before_ek7c`). Kuru sınama (sahte koşular, geçici klasör) 16/16 geçti: ×2,0 → ×2,8 kenar, S2/S3 ayarlı taban dalları, iki tabanlı raporlar; ayarlı taban gerekmeyen senaryoda ek koşu yok.

<!-- K6-SONUC-BAS -->
## Kuyruk6 sonuçları (Ek 7 kapılarına göre; otomatik, sonuçlar GÖRÜLDÜKTEN SONRA yazıldı; yorum devir notunda)

- F0: F0 (T1): en iyi taban ayarı LR×2.0 WD 0.1 · g4 kayıp 3.5943 (eski taban 3.6552, eşik 3.6187) ⇒ K0 TUTTU: ayarlı taban; m’ler iki tabana göre yazılır; paket m₄: t1 eski 2.207 / ayarlı 1.795 (Haiku'ya +5.08%); t2 eski 2.185 / ayarlı 1.778 (Haiku'ya +5.00%).
- F1A: F1A: geçen kollar ['buyuyen', 'muonh+buyuyen']; boyla büyüme işareti (a₄ ≥ a₃) ['muonh+buyuyen']; tutulan bayraklar yok. muonh a₄ 0.917; buyuyen a₄ 1.153; muonh+buyuyen a₄ 1.101; overtok a₄ 1.032; hub4 a₄ 1.023; derin a₄ 1.033.
- BIRLESIM: BİRLEŞİM: paket2 canon+muonh+qknorm+bigram+wsd+buyuyen (LR×2.8); g4 m 2.458 / 2.452 (ort 2.455, Haiku'ya +7.80%) ⇒ YENİ EN İYİ TARİF.
- TOHUM2: TOHUM2: muonh+buyuyen: tohum 2 a₃ 1.065 a₄ 1.118 ⇒ İKİ TOHUMLU İŞARET.
- F1B: F1B: paket − taban (OOD ort, puan): b1 +2.2, b2 +6.9, b3 +7.5 ⇒ **GEÇTİ**; dongu − taban (OOD ort, puan): b1 +1.1, b2 +8.4, b3 +4.8 ⇒ **geçmedi (kazanç < 5 puan ya da büyük boyda küçüldü)**; b1 dongu test anı turu 1 → 4: OOD +11.9 puan ⇒ TEST ANI İŞARETİ; b2 dongu test anı turu 1 → 4: OOD +22.6 puan ⇒ TEST ANI İŞARETİ; b3 dongu test anı turu 1 → 4: OOD +15.1 puan ⇒ TEST ANI İŞARETİ; self-consistency (8 örnek, çoğunluk) b3 paket: OOD 0.482 → 0.490.
- F1A2: F1A-2: gptbert a₄ 0.600 ⇒ ÖLDÜ; rholoss a₄ 0.540 ⇒ ÖLDÜ. T2 (4 tur × ¼ veri; kontrol): taban+ep4 g3: m 0.901 (tek tura göre oran 0.901); taban+ep4 g4: m 0.878 (tek tura göre oran 0.878); canon+muonw+qknorm+bigram+wsd+ep4 g3: m 2.275 (tek tura göre oran 0.464); canon+muonw+qknorm+bigram+wsd+ep4 g4: m 1.264 (tek tura göre oran 0.573).
- F1K: F1K: | tek | taban | 1.310M | 0.740 | 0.526 | 0.526 | 0.225 | | tek | paket | 2.502M | 1.408 | 1.002 | 0.090 | 0.857 | | cok5 | taban | 1.507M | 0.851 | 0.605 | 0.605 | 0.331 | | cok5 | paket | 2.648M | 1.490 | 1.061 | 0.096 | 0.999 | taban: ifade çeşitliliği (cok5 − tek) QA +10.6 puan ⇒ veri yoğunlaştırma yararlı; paket: ifade çeşitliliği (cok5 − tek) QA +14.2 puan ⇒ veri yoğunlaştırma yararlı; Literatür: ≈2 bit/param ≈1000 görülmede, ≈1 bit/param 100 görülmede (Allen-Zhu & Li 2024, sentetik veri)..
- F2: F2 (S1 25M): paket m 1.995 (Haiku'ya +6.00%); ayarlı tabana (LR ×2.8) göre 1.538 (Haiku'ya +3.74%), c_S1 1.297 · en iyi η 0.008, büyütme LR 2.0 · K2'yi geçen ['S1_w_p2', 'S1_b2.0'].
- F3: F3 (S2 49M): paket m 2.429 (Haiku'ya +7.71%); - S2_w_p2: r(S1) 1.306 → r(S2) 1.299 ⇒ büyüme yok; - S2_b2.0: r(S1) 1.217 → r(S2) 1.159 ⇒ büyüme yok. Ek 7c: ayarlı tabana göre paket m 1.922 (Haiku'ya +5.68%), c_S2 1.263.
- F4: F4 (S3 85M kanıt): bpb 0.9709 ⇒ BANT İÇİ (yasa tuttu); aktarım m 2.765 ⇒ KORUNDU (≥2,0) (Haiku'ya +8.83%). Ek 7c: ayarlı S3 tabanına göre m 2.212 ⇒ KORUNDU (≥2,0) (Haiku'ya +6.89%).
- F5: F5: otomatik toplu rapor yazıldı (Devrim kökünde RAPOR_DEVRIM_OTOMATIK.md); anlatı raporu ve İngilizce özet Claude'un işi.
- F0 hükmü (t1_hukum.json): ayarli=True; esik=3.6186578999999996; eski_g4=3.65521; g4=3.59433; zaman=2026-10-12 15:43; ayarlı g2 kayıp 4.7911; ayarlı g3 kayıp 4.1443; ayarlı g4 kayıp 3.5943
<!-- K6-SONUC-SON -->

## Ek 8 — hız: canon kaydırmalı toplam (13 Ekim 2026 ~15:20, Claude; Mert'in “hiçbir şeyi bozmadan hızlandır” isteği)

- Değişiklik: `merdiven.Blok._canon` için `MERDIVEN_HIZLI_CANON=1` bayrağı (varsayılan kapalı; kapalıyken yol bit bit aynı): nedensel derinlik evrişimi (pencere 4) cuDNN evrişimi + transpoze yerine kaydırmalı toplamla hesaplanır; matematik aynı, derleyici tek çekirdekte birleştirir. Deney tasarımı, kollar, bütçeler, eşikler değişmez.
- Ölçüm: paket (canon+muonw+qknorm+bigram+wsd), L8 d512, 40 adım, derlemeli, adım 20–39 penceresi: bayrak kapalı 60.031 ve 60.109 jeton/sn, açık 66.297 ⇒ +%10,4; kayıp farkı açık−kapalı en büyük 0,0023 (son 10 adım ort. 0,0014), aynı kodun tekrarı 0,0019 (0,0014) ⇒ fark gürültü tabanında (derleme sınamasının ölçütü |fark| < 0,02); CPU fp32 sınaması ileri 4,8e-7, türev 7,6e-6. Ayrıntı `HIZ_OPTIMIZASYON\olcum_canon_*.json`.
- Uygulama: F3 ve F4 komutları (`k6_buyuk.derle_env`). F2 başlamış faz olduğu için kalan kolları eski yolla koşar (süreç eski fonksiyonu bellekte tutar); kuyruk yeniden başlarsa F2'nin kalanı da hızlı yolla koşar, eşdeğerlik ölçüldüğü için sonuç etkilenmez.
- Etki: m-hesap FLOP tabanlı, değişmez; m-süre canonlu kollarda ≈%10 iyileşir (raporda not edilir). Tabanda canon yok, hızı aynı.
- Not (15:23): ölçüm için yapılan duraklatma, F2–F4'ün devam düzeneğinde eski bir hatayı ortaya çıkardı: `torch.load(map_location='cuda')` CPU RNG durumunu GPU'ya taşıyordu, `torch.set_rng_state` çöktü; S1_h0.004 (105. adımda duraklatılmıştı) 3 denemede düştü (HATA) ve kuyruk kuralı gereği sıradaki kola (S1_h0.016) geçti. Düzeltme 15:2x: `egit_evren.py` ve `egit_h.py`'de `set_rng_state(k["rng"].cpu())` (matematik aynı; yedekler `.bak_before_rng`; sentetik sınama geçti). F2'nin η taraması eksik noktayı atlayarak sürer (`v = {..} if sonuc(..)`); en iyi η 0,008 çıkarsa 0,004 kenar kontrolü eksik kalır, F5 raporunda sınır olarak yazılır. Bu hata düzeltilmeseydi laptopu götürmek için yapılacak her duraklatma F2–F4'te aynı şekilde çökecekti.
