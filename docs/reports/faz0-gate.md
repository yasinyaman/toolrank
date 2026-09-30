# Faz 0 — Kapı raporu (29 Eyl 2026)

Durum: karar verildi ve onaylandı (29 Eyl 2026). Çekirdek adapter **Qwen3-Embedding-8B + kendi skip
head'lerimiz**. Kapı tutmadı; takvim değişmiyor, kapsam daralıyor (aşağıda "Faz 0'ın kalan
maddeleri"). Üç set de koşuldu: ToolRet (kapı), LiveMCPBench ve MCP-Zero (held-out MCP setleri).
Sayılar `faz0-week2.md`, `faz0-week3.md` ve `faz0-week4.md`'deki koşulardan.

## Sonuç (tek cümle)

Kapı tutmadı: fine-tune'lu CLM kapı ayarında (ToolRet, w/ inst, cat-macro NDCG@10) 17.68'de kaldı
(kapı ≥ 50); aynı yöntemle Qwen3-Embedding-8B üzerine eğitilen head'ler ~47.2 ile en iyi sonucu
veriyor (veri 60K'dan 206K çifte çıkınca da değişmiyor) ve iki held-out MCP setinde de
(LiveMCPBench, MCP-Zero) önde — karar, çekirdek adapter'ı "Qwen3-Embedding + kendi head'lerimiz"
yapmak.

## Tablo

ToolRet (44.453 tool, 7.961 sorgu), w/ inst, NDCG@10 ve Comprehensiveness@10; cat-macro kapının
okunduğu sütun (makalenin ve StackOne'ın "Average"ı):

| Scorer | NDCG@10 micro | NDCG@10 cat-macro | C@10 cat-macro |
| --- | ---: | ---: | ---: |
| BM25 (makale ayarı, `--no-stem`) | 39.27 | 36.41 | 39.03 |
| CLM sıfır-atış (`example_call`) | 6.64 | 4.99 | 6.51 |
| CLM fine-tune (60K çift) | 28.88 | 17.68 | 19.70 |
| Qwen3-Embedding-8B sıfır-atış (`documentation`) | 51.11 | 46.54 | 47.64 |
| Qwen3-Embedding-8B + head (60K çift) | 52.55 | 47.23 | 47.98 |
| Qwen3-Embedding-8B + head (tüm veri, 206K çift) | 54.03 | 47.14 | 47.50 |
| StackOne v2 (bildirilen, aynı ayar) | — | 54.4 | — |

LiveMCPBench (525 MCP tool'u, 94 görev), w/ inst; sorgu görevin tam metni, gold görevde kullanılan
tool'lar (ortalama 2.7):

| Scorer | Recall@5 micro | Recall@5 cat-macro | NDCG@10 micro |
| --- | ---: | ---: | ---: |
| BM25 | 20.33 | 18.50 | 21.75 |
| BM25, w/o inst (genel talimat BM25'i düşürüyor) | 28.95 | 26.72 | 29.29 |
| CLM sıfır-atış | 4.56 | 5.95 | 4.96 |
| CLM fine-tune (60K, ToolRet-train) | 8.57 | 11.47 | 6.79 |
| Qwen3-Embedding-8B sıfır-atış | 49.04 | 49.64 | 51.42 |
| Qwen3-Embedding-8B + head (60K, ToolRet-train) | **52.25** | 51.84 | **54.65** |
| Qwen3-Embedding-8B + head (tüm veri, ToolRet-train) | 52.05 | **52.01** | 52.84 |

Planın ikincil hedefi LiveMCPBench Recall@5 ≥ 0.83; o sayının protokolü belli değil. Bu protokolde
6 tool'luk bir görev top-5'e sığmadığı için Recall@5'in tavanı 1'in altında; 0.83 büyük ihtimalle
adım başına tek tool sorgulanan bir kurgudan. Karşılaştırılabilir değil.

MCP-Zero (MCP-tools: 308 sunucu, 2.792 tool), tool başına bir sorgu. Makalenin promptlarıyla bir
LLM'in (burada Qwen3-8B) yazdığı `server: … / tool: …` isteği sorgu oluyor. Precision@1, makalenin
top-1 isabeti; tüm korpusta düz retrieval (makale hiyerarşik eşleşiyor, tek sayı vermiyor):

| Scorer | inst | Precision@1 | Recall@5 | NDCG@10 |
| --- | :-: | ---: | ---: | ---: |
| BM25 | y | 21.60 | 49.22 | 39.15 |
| BM25 | n | 56.34 | 76.30 | 68.83 |
| CLM sıfır-atış | y | 1.47 | 4.48 | 3.63 |
| CLM fine-tune (60K, ToolRet-train) | y | 4.62 | 11.14 | 9.68 |
| Qwen3-Embedding-8B sıfır-atış | y | 69.73 | 89.28 | 82.01 |
| Qwen3-Embedding-8B + head (60K, ToolRet-train) | y | 71.38 | 90.33 | 83.23 |
| Qwen3-Embedding-8B + head (tüm veri, ToolRet-train) | y | **71.99** | **91.30** | **83.77** |

Tool metnine sunucu adı eklenince (ablasyon, `faz0-week4.md`) top-1 Qwen3 + head'te 79.87'ye,
BM25'te 80.44'e çıkıyor; LiveMCPBench'te aynı değişiklik Recall@5'e +1 ile +3 arası ekliyor (Qwen3
+ head 60K: 53.72). Head'lerin Qwen3'e göre farkı iki ayarda da korunuyor.

## Gecikme

GB10, batch 1, tüm ToolRet indeksi (44k tool), w/ inst sorgular (`scripts/latency.py`):

| Scorer | Soğuk p50 ms | Soğuk p95 ms | Sıcak p50 ms |
| --- | ---: | ---: | ---: |
| CLM, Qwen3-8B bf16 | 85.9 | 104.0 | 1.8 |
| CLM, Qwen3-8B FP8 | 49.3 | 59.1 | 1.8 |
| Qwen3-Embedding-8B bf16 | 93.7 | 115.2 | 7.4 |

Soğuk gecikmenin neredeyse tamamı 8B modelin tek ileri geçişi (bellek bant genişliği sınırlı);
FP8 bunu yarıya indiriyor ve head çıktılarını bozmuyor (kosinüs ≥ 0.997). Qwen3-Embedding'in FP8
kopyası ölçülmedi; aynı nedenle benzer bir düşüş beklenir. İkincil hedef (p50 ≤ 100 ms) iki
backbone'da da tutuyor. Skip head'ler 4096 boyutta kaldığı için sıcak skor Qwen3'ün ham skoruyla
aynı sınıfta (~7 ms); 512 boyutlu CLM head'leri bunu ~2 ms'ye indiriyor.

## Maliyet (GB10 saat)

Tek makine (GB10), duvar saati:

| İş | Süre |
| --- | ---: |
| ToolRet korpusu, 24 koşuluk sıfır-atış matris (iki backbone, dört format) | 2 sa 15 dk (35 dk vLLM takılması dahil) |
| ToolRet-train gömme, 60K + 2K çift, iki backbone aynı anda | 1 sa 7 dk |
| ToolRet-train gömme, kalan 146K çift, Qwen3-Embedding | ~1 sa |
| Head eğitimi, 23 koşu (epoch süresi; vektörler cache'ten) | 25 dk |
| Gecikme, FP8, LiveMCPBench | ~30 dk |
| MCP-Zero: indirme, üretim modelinin açılışı, 2.792 sorgu üretimi (~3 dk), 18 değerlendirme koşusu (ablasyonlar dahil) | ~15 dk |
| **Toplam** | **~5.5 sa** |

Head eğitimi ucuz (60K çift × 5 epoch ~1 dk); maliyetin tamamı bir kerelik gömme. Embedding cache
şu an 8+ GB.

## Karar ve gerekçe

Planın seçenekleri: CLM devam / Qwen3-Embedding + kendi head'lerimiz / hibrit. Karar (29 Eyl 2026,
onaylandı): **Qwen3-Embedding + kendi head'lerimiz.**

- **CLM devam: seçilmedi.** Sıfır-atışta 4.99; referans head'lerden başlayan fine-tune 17.68'e
  çıkarıyor ama veri 15K'dan 60K'ya çıkınca cat-macro ~17'de kalıyor. Kazanç ToolRet'in büyük
  görevlerinde; LiveMCPBench'e taşınmıyor (Recall@5 4.6 → 8.6). Ham Qwen3-8B vektörlerinde doğrudan
  kullanılabilir retrieval sinyali yok (sıfırdan head'ler öğrenmiyor). MCP-Zero'da top-1 1.5, fine-tune
  sonrası 4.6.
- **Qwen3-Embedding + kendi head'lerimiz: seçildi.** Sıfır-atışta StackOne'ın Qwen3 sayısını
  yeniden üretiyor (46.54 / 46.2). Kimlikten başlayan skip head'ler başlangıcı koruyarak ToolRet'te
  micro'yu 51.11'den 54.03'e çıkarıyor (cat-macro ~47.2'de kalıyor: kazanç eğitimde sık görülen
  büyük görevlerde) ve alan dışında LiveMCPBench Recall@5'i 49.0'dan ~52'ye, MCP-Zero top-1'i
  69.7'den 72.0'a taşıyor: üç sette de aynı yönde, küçük ama tutarlı. Gecikme aynı sınıfta, head
  eğitimi dakikalar.
- **Hibrit: seçilmedi.** CLM'in sıralama kalitesi bu kadar düşükken reranker olarak da değer katması
  beklenmez.

Kapı (≥ 50) tutmadığı için planın kuralı geçerli: takvim değişmez, kapsam daralır. README'nin
cümlesiyle: fine-tune'dan sonra CLM Qwen3-Embedding'i geçemedi, çekirdek adapter değişir, yol
haritasının geri kalanı değişmez. Seçilen yol da kapı eşiğinin (50) altında (cat-macro 47.2);
eşiği bu yolla geçmek Faz 1'in çıkış kriteri değil, backlog'daki kaldıraçlarla (aşağıda) ölçülür.

CLM adapter'ı ve checkpoint yükleyicisi repo'da kalıyor: skip head'ler aynı formatı kullanıyor ve
`--scorer clm` bir benchmark adapter'ı olarak ölçülebilir kalıyor.

## Faz 0'ın kalan maddeleri

- Sentetik "örnek çağrı" üretimi (Hafta 2): CLM'in `example_call` formatı içindi; backlog'a "LLM
  ile tool metni zenginleştirme" olarak taşındı (`ChatModel` portu hazır).
- Replay (Hafta 3): CLM head'lerinin unutmasına karşıydı; düştü. Skip head'ler kimlikten başladığı
  için ayrı bir replay'e gerek yok.
- Ablasyonlar (Hafta 3): negatif kaynağı ve filtresi, lr, veri boyu hafta 3'te ölçüldü; kalan
  batch boyu ve temiz negatifler backlog'da.
- Qwen3-Embedding-8B LoRA (Hafta 4, "zaman kalırsa"): backlog'da.

## Açık kalanlar

- **Eşik için sıradaki kaldıraçlar (backlog):** daha fazla veri değil (60K → 206K çift cat-macro'yu değiştirmedi);
  daha temiz negatifler (mined negatifler Qwen3'ü 10 puana kadar düşürüyor; eşdeğer tool'ları LLM ile
  ayıklamak), görev dengesini gözeten örnekleme (kazanç büyük görevlerde kalıyor), Qwen3-Embedding
  LoRA fine-tune'u (planda Hafta 4'ün opsiyonel maddesi).
- **Doğrulama seti.** Eğitim çiftlerinden alınan doğrulama kapı metriği için vekil değil (mined
  negatiflerle val artarken benchmark düşüyor). Faz 1'deki head güncellemeleri için benchmark'a
  benzeyen, test setinden bağımsız bir geliştirme seti gerekiyor.
- **Anotasyon ve örtüşme.** ToolRet sorgu başına tek gold küme taşıyor ve test gold tool'larının %36'sı
  eğitimde pozitif; fine-tune kazançları bu iki etkiyle birlikte okunmalı.

## Faz 1'e etkiler

`docs/plan/faz-1.md` ve `faz-2.md` buna göre güncellendi; tarih değişmedi.

- Faz 1 Hafta 2'deki "Faz 0'ın kazanan head'ini paketle" maddesi CLM checkpoint'i yerine
  Qwen3-Embedding-8B + skip head olur (`toolrank eval --clm-ckpt` ile yüklenen aynı format; iki
  head toplam 29.9M parametre).
- Varsayılan dağıtım için FP8 backbone adayı. CLM backbone'unda (Qwen3-8B) kalite aynı kaldı,
  gecikme yarıya indi. Qwen3-Embedding-8B'nin FP8 kopyası ölçülmedi, paketlemeden önce ölçülür.
- MCP tool'ları sunucu adıyla birlikte indekslenmeli (ingestion / MCP proxy). MCP-Zero'da top-1'e
  +8 (Qwen3 + head) ve +24 (BM25) puan, LiveMCPBench'te +1–3 ekliyor ve hiçbir skorer'da zarar
  vermiyor. Sunucu önce, tool sonra gelen hiyerarşik eşleşme (Faz 2) bunun bir adım ötesi.
- LiveMCPBench ve MCP-Zero'da genel bir talimat BM25'i düşürüyor (sorgular kısa, talimattaki
  sözcükler tool metinlerinde sık). Hibrit/BM25 yolunda talimat sorguya eklenmemeli.
- Faz 2'nin öğrenme döngüsü (kullanım günlüğü → hard negative → gecelik head eğitimi) için ders:
  madenlenmiş negatifler kaliteyi düşürebiliyor; eşdeğer tool kontrolü ve benchmark'a benzeyen bir
  doğrulama olmadan head güncellemesi yayınlanmamalı.
- Faz 2'nin hiyerarşik routing maddesindeki CLM'e özgü parçalar (`Choice` ile sunucu ön kararı,
  `Noul` kapısı) adapter'dan bağımsız karşılıklarıyla değişti: sunucu açıklaması ve özetiyle
  eşleşme, skor eşiği.
