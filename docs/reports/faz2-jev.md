# Faz 2 — Jev, CLM, cross-encoder ve LoRA karşılaştırması (30 Eyl – 1 Eki 2026)

TypeSafe AI'ın 15 Eylül 2026'da duyurduğu "System One" modeli Jev'den yola çıkan soru: toolrank'in
kısa listesinin üstüne, isteği ve adayı birlikte okuyan bir ikinci aşama koymak ne kazandırır, ve
bunu yerelde ne karşılar? Aynı protokol, aynı setler, aynı araç metniyle ölçüldü: Jev (barındırılan),
CLM_v0.1-8B (Faz 0'ın çift kodlayıcısı), Qwen3-Reranker-8B ve bge-reranker-v2-gemma (yerel
cross-encoder'lar) ve omurganın kendisinin LoRA ile eğitimi. Plan maddesi değil; Faz 2'nin 1. haftası.

## Sonuç (tek cümle)

Omurgayı LoRA ile eğitmek, tek aşamada ve ek gecikme olmadan, ToolRet'te hem head'leri hem ikinci
aşamalı düzenleri geçiyor (NDCG@10 54.03 → 58.90, cat-macro 47.13 → 54.36: Faz 0 kapısının 50 eşiğinin
üstü, StackOne v2'nin 54.4'ü düzeyi) ve head'ler onun üstüne bir şey katmıyor; hiç görülmemiş
LiveMCPBench'te ise LoRA +2 puanda kalırken kısa listeyi isteğiyle birlikte okuyan bir ikinci aşama
+9–12 puan veriyor, ve bunu barındırılan Jev ile yerel Qwen3-Reranker-8B başa baş yapıyor (heads'in
ilk 20'si + dokümantasyon: ToolRet 57.69 / 58.05, LiveMCPBench 64.03 / 62.68, MCP-Zero top-1 92.34 /
91.26); çift kodlayıcı CLM aynı koltukta listeyi bozuyor, bge-reranker-v2-gemma MCP setlerinde
yetmiyor.

## Özet

İlk iki satır tek aşama (yalnız gömme modeli); sonrakiler kısa listenin üstünde bir ikinci aşama.
"Sorgu başına bedel" ikinci aşamanın eklediği gecikme, ücret ve bellek. Bu tablo ve aşağıdaki tam
tablolar `scripts/rerank_report.py --write` ile sonuç dosyalarından üretiliyor.

<!-- summary:start -->
| Düzen | ToolRet NDCG@10 ↑ % | ToolRet cat-macro ↑ % | LiveMCPBench NDCG@10 ↑ % | MCP-Zero Precision@1 ↑ % | sorgu başına bedel |
|---|---:|---:|---:|---:|---|
| heads (Qwen3-Embedding-8B + v0.1 head'leri), tek aşama | 54.03 | 47.13 | 53.95 | 79.87 | ~1 ms, yerel |
| Qwen3-Embedding-8B + LoRA, tek aşama | 58.90 | 54.36 | 55.74 | 88.57 | ~1 ms, yerel; MCP-Zero seçim seti |
| heads → Jev, ilk 20 + dokümantasyon | 57.69 | 52.71 | 64.03 | 92.34 | +0.3 s, 0.0002 $, dış API |
| heads → Jev, ilk 100 | 55.04 | 51.41 | 66.25 | 91.55 | +0.3 s, 0.0002 $, dış API |
| heads → Qwen3-Reranker-8B, ilk 20 + dokümantasyon | 58.05 | 52.93 | 62.68 | 91.26 | +0.3–0.6 s, yerel, +16 GB |
| heads → bge-reranker-v2-gemma, ilk 20 + dokümantasyon | 53.96 | 50.26 | 36.19 | 48.24 | +0.1 s, yerel, +5 GB |
| heads → CLM-8B, ilk 20 | 28.94 | 24.80 | 27.65 | 10.10 | +2 ms, yerel |
| zero-shot Qwen3-Emb → Jev, ilk 20 + dokümantasyon | 55.98 | 52.67 | 62.48 | 91.98 | +0.3 s, dış API |
| Jev tek başına, parçalı | — | — | 65.05 | 90.04 | 0.3–0.6 s, 40–80k token, dış API |
<!-- summary:end -->

## Ölçüler, birimler ve yön

Her tablo başlığında ok yönü (↑ yüksek iyi, ↓ düşük iyi) ve birim var. Tüm sıralama ölçüleri
ToolRet protokolüyle (tüm külliyat üstünde top-100, sorgular üstünde mikro ortalama) hesaplanıyor
ve yüzde olarak yazılıyor; "fark" sütunları yüzde puanı.

| Ölçü | Ne | Birim | İyi olan |
|---|---|---|---|
| NDCG@k | ilk k'daki sıralama kalitesi; doğru araç ne kadar üstteyse o kadar yüksek | % (0–100) | ↑ |
| Recall@k | doğru araçların ilk k içinde bulunan payı | % | ↑ |
| Precision@1 | ilk sıradaki aracın doğru olduğu sorguların payı; MCP-Zero'nun "top-1 doğruluğu" | % | ↑ |
| Comprehensiveness@k | bütün doğru araçları ilk k içinde bulunan sorguların payı | % | ↑ |
| cat-macro | NDCG@10'un önce ToolRet kategorisi (web / code / customized) içinde görevler üstünde, sonra kategoriler üstünde düz ortalaması; kâğıdın "Average"ı | % | ↑ |
| sorgu p50 | eval'in `rank` adımının sorgu başına medyan süresi; Jev ve cross-encoder satırlarında 8 eşzamanlı isteğin toplam süresi bölü sorgu sayısı, yani verim | ms | ↓ |
| çağrı p50 | tek bir Jev ya da skor API çağrısının medyan süresi, GB10'dan | ms | ↓ |
| token / sorgu | Jev'e gönderilen, faturalanan girdi tokenı, sorgu başına | token | ↓ |
| çift / sorgu | cross-encoder'ın puanladığı (istek, aday) çifti sayısı, sorgu başına | adet | ↓ |
| ücret | satırın tamamı, 0.042 $ / M token ile | $ | ↓ |
| parite kosinüsü | süreç içi vektör ile servisten gelen vektörün kosinüs benzerliği; 1 = aynı | 0–1 | ↑ |
| loss | InfoNCE eğitim kaybı | nat | ↓ |

## Jev nedir, buraya nasıl oturur

- Jev metin üretmiyor: bir "state" ve tipli sorular alıyor; bir Choice sorusu en fazla 255 seçenek
  için olasılık dağılımı döndürüyor. Girdi tokenı 0.042 $ / M, çıktı ücretsiz; istek başına 64k,
  state + en uzun soru 32k token; 100K token/s, 40 istek/s. Model `jev-1.13.0` (alias'lar kayar).
- 255 sınırı yüzünden 44 bin araçlık ToolRet'i tek başına tarayamaz. Kendi dokümanlarının önerisi
  "önce hızlı arama, sonra yeniden sıralama"; skill suggestion cookbook'u 182 skill'i tek Choice ile
  sıralayıp ilk 3'ü tekrar okuyor.
- toolrank'te iki adapter: `--rerank jev` herhangi bir scorer'ın ilk `--rerank-depth` aracını tek
  Choice ile yeniden sıralar (olasılık = skor, eşitlikte taban sıra, derinliğin altı taban sırayla
  devam eder, top-100 protokolü bozulmaz); `--scorer jev` Jev'i tek başına koşar (`--jev-chunk`
  büyüklüğünde parçalar, parça kazananları bir kez daha sıralanır). State `{"request"}`, kıyaslama
  talimatı sorunun başında, seçenek metni `--jev-tool-format` (`--jev-max-chars` ile kesilir).
- Cevaplar istek gövdesiyle anahtarlanıp `.cache/toolrank/jev.sqlite`'a yazılıyor: tekrar koşu
  ücretsiz ve birebir aynı sıralama. GB10'daki önbellekte 36.907 cevap var.

## Kollar

| Kol (koşuldu mu) | ToolRet | LiveMCPBench | MCP-Zero |
|---|---|---|---|
| BM25 (w/ inst), Qwen3-Embedding-8B, heads | README satırları | README satırları | README satırları |
| heads ilk 100 → Jev, name_desc, 1000 karakter | evet | evet | evet |
| heads ilk 20 → Jev, documentation, 3000 karakter | evet | evet | evet |
| BM25 ilk 30 → Jev, name_desc (cookbook düzeni) | evet | evet | evet |
| Jev tek başına, parça 200, parça başı 20 kazanan | hayır (44k araç) | evet (3 parça) | evet (14 parça) |
| heads ilk 100 ve ilk 20 → CLM_v0.1-8B (8090, example_call, `clm` sorgu formatı) | evet | evet | evet |
| heads ilk 100 → fine-tune'lu CLM (`clm_60k_lr1e-2.pt`) | evet | evet | evet |
| BM25 ilk 30 → CLM_v0.1-8B | evet | evet | evet |
| aynı üç satır, Jev'in okuduğu metinle: name_desc 1000 karakter, documentation 3000 karakter (`--rerank-max-chars`) | evet | evet | evet |
| heads ilk 20 → Qwen3-Reranker-8B ve bge-reranker-v2-gemma, Jev'in metniyle | evet | evet | evet |
| zero-shot Qwen3-Emb ilk 100 ve ilk 20 → Jev | evet | evet | evet |
| Qwen3-Embedding-8B + LoRA (20 bin çift), tek aşama; üstüne head eğitimi | evet | evet | evet (seçim seti) |
| heads ilk 100 / BM25 ilk 30 → cross-encoder'lar; cross-encoder tek başına; LoRA → Qwen3-Reranker | koşuyor | koşuyor | koşuyor |

## Tablo

w/ inst; taban satırları `docs/results/readme_*.json` (README tablosu), Jev satırları
`results/jev_*.json`, CLM satırları `results/clm_*.json` (`scripts/clm_rerank.sh`, `--rerank clm`).
"sorgu p50 ms" eval'in `rank` adımı: Jev satırlarında 8 eşzamanlı isteğin toplam süresi bölü sorgu
sayısı, yani verim, çağrı başı gecikme yandaki sütunda; CLM satırlarında GPU'daki head'ler ve
önbellek aramaları. Son üç sütun Jev'e ait: GB10'dan çağrı başına p50, sorgu başına faturalanan token
ve satırın toplam ücreti, 0.042 $ / M token ile.

<!-- tables:start -->
### ToolRet (w/ inst, n=7961)

| Satır | NDCG@10 ↑ % | Recall@5 ↑ % | Recall@10 ↑ % | Comprehensiveness@10 ↑ % | cat-macro ↑ % | sorgu p50 ↓ ms | çağrı p50 ↓ ms | token ya da çift / sorgu ↓ | ücret ↓ $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 39.27 | 42.42 | 49.49 | 40.70 | 36.41 | 0.4 | — | — | — |
| BM25 → Jev, ilk 30 | 47.81 | 51.56 | 56.30 | 46.72 | 45.19 | 39.8 | 285 | 1,965 token | 0.66 |
| Qwen3-Embedding-8B | 51.11 | 54.24 | 62.32 | 51.60 | 46.54 | 0.8 | — | — | — |
| heads | 54.03 | 57.39 | 65.59 | 55.06 | 47.13 | 0.9 | — | — | — |
| heads → Jev, ilk 100 | 55.04 | 58.25 | 66.76 | 56.81 | 51.41 | 43.1 | 310 | 5,564 token | 1.86 |
| heads → Jev, ilk 20, documentation | 57.69 | 61.56 | 68.35 | 58.30 | 52.71 | 42.0 | 303 | 4,047 token | 1.35 |
| Qwen3-Emb → Jev, ilk 100 | 54.59 | 58.04 | 66.03 | 55.90 | 52.05 | 45.0 | 320 | 5,459 token | 1.83 |
| Qwen3-Emb → Jev, ilk 20, documentation | 55.98 | 59.55 | 65.41 | 54.80 | 52.67 | 43.2 | 306 | 4,138 token | 1.38 |
| heads → CLM-8B, ilk 100 | 15.36 | 17.25 | 25.56 | 20.31 | 12.47 | 3.6 | — | — | — |
| heads → CLM-8B, ilk 20 | 28.94 | 31.37 | 48.72 | 38.61 | 24.80 | 1.9 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 34.20 | 36.50 | 46.57 | 37.77 | 22.68 | 3.2 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 22.33 | 25.39 | 36.33 | 29.83 | 19.59 | 1.7 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 13.84 | 15.26 | 23.61 | 18.62 | 11.03 | 3.8 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 28.65 | 31.21 | 48.09 | 38.49 | 23.89 | 7.0 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 21.08 | 23.63 | 34.86 | 28.64 | 18.56 | 1.8 | — | — | — |
| heads → Qwen3-Reranker-8B, ilk 20, documentation | 58.05 | 61.80 | 68.41 | 58.12 | 52.93 | 613.3 | 4517 | 20 çift | — |
| heads → bge-reranker-v2-gemma, ilk 20, documentation | 53.96 | 57.80 | 66.99 | 56.21 | 50.26 | 231.6 | 1888 | 20 çift | — |
| Qwen3-Emb + LoRA | 58.90 | 62.90 | 69.54 | 60.04 | 54.36 | 19.9 | — | — | — |
| Qwen3-Emb + LoRA + head (epoch 0 = kimlik) | 58.90 | 62.90 | 69.54 | 60.04 | 54.36 | 7.7 | — | — | — |

### LiveMCPBench (w/ inst, n=94)

| Satır | NDCG@10 ↑ % | Recall@5 ↑ % | Recall@10 ↑ % | Comprehensiveness@10 ↑ % | sorgu p50 ↓ ms | çağrı p50 ↓ ms | token ya da çift / sorgu ↓ | ücret ↓ $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 25.38 | 22.92 | 31.41 | 17.02 | 0.3 | — | — | — |
| BM25 → Jev, ilk 30 | 39.47 | 38.09 | 44.07 | 22.34 | 11.0 | 298 | 3,438 token | 0.01 |
| Qwen3-Embedding-8B | 53.74 | 50.82 | 61.09 | 37.23 | 0.4 | — | — | — |
| heads | 53.95 | 53.03 | 61.66 | 36.17 | 0.1 | — | — | — |
| heads → Jev, ilk 100 | 66.25 | 65.84 | 73.68 | 46.81 | 12.2 | 329 | 7,018 token | 0.03 |
| heads → Jev, ilk 20, documentation | 64.03 | 64.70 | 69.63 | 44.68 | 11.7 | 313 | 4,984 token | 0.02 |
| Jev tek başına | 65.05 | 66.44 | 73.81 | 45.74 | 49.5 | 360 | 40,910 token | 0.16 |
| Qwen3-Emb → Jev, ilk 100 | 66.00 | 66.15 | 74.99 | 51.06 | 46.1 | 327 | 7,010 token | 0.03 |
| Qwen3-Emb → Jev, ilk 20, documentation | 62.48 | 63.99 | 68.81 | 43.62 | 45.6 | 315 | 5,062 token | 0.02 |
| heads → CLM-8B, ilk 100 | 11.85 | 12.36 | 19.79 | 8.51 | 5.5 | — | — | — |
| heads → CLM-8B, ilk 20 | 27.65 | 29.27 | 43.90 | 15.96 | 3.9 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 13.49 | 11.23 | 21.22 | 10.64 | 5.0 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 15.07 | 15.47 | 22.91 | 9.57 | 4.0 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 10.93 | 10.76 | 18.46 | 9.57 | 135.0 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 21.06 | 16.98 | 36.90 | 14.89 | 323.1 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 13.48 | 12.75 | 22.65 | 9.57 | 4.1 | — | — | — |
| heads → Qwen3-Reranker-8B, ilk 20, documentation | 62.68 | 62.84 | 71.68 | 47.87 | 198.2 | 1667 | 20 çift | — |
| heads → bge-reranker-v2-gemma, ilk 20, documentation | 36.19 | 33.71 | 50.45 | 26.60 | 51.5 | 1938 | 20 çift | — |
| Qwen3-Emb + LoRA | 55.74 | 52.06 | 63.34 | 39.36 | 36.5 | — | — | — |
| Qwen3-Emb + LoRA + head (epoch 0 = kimlik) | 55.74 | 52.06 | 63.34 | 39.36 | 4.1 | — | — | — |

### MCP-Zero (w/ inst, n=2792)

| Satır | NDCG@10 ↑ % | Recall@5 ↑ % | Recall@10 ↑ % | Comprehensiveness@10 ↑ % | Precision@1 ↑ % | sorgu p50 ↓ ms | çağrı p50 ↓ ms | token ya da çift / sorgu ↓ | ücret ↓ $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 63.32 | 73.94 | 84.95 | 84.78 | 45.63 | 0.1 | — | — | — |
| BM25 → Jev, ilk 30 | 92.84 | 94.55 | 94.89 | 94.70 | 90.29 | 39.1 | 286 | 1,289 token | 0.15 |
| Qwen3-Embedding-8B | 87.21 | 92.31 | 95.63 | 95.52 | 78.19 | 0.1 | — | — | — |
| heads | 88.53 | 94.20 | 96.12 | 96.02 | 79.87 | 0.1 | — | — | — |
| heads → Jev, ilk 100 | 94.90 | 96.96 | 97.55 | 97.46 | 91.55 | 41.0 | 292 | 3,104 token | 0.36 |
| heads → Jev, ilk 20, documentation | 95.01 | 96.55 | 97.23 | 97.13 | 92.34 | 38.8 | 285 | 1,796 token | 0.21 |
| Jev tek başına | 93.95 | 96.61 | 96.81 | 96.67 | 90.04 | 629.7 | 320 | 90,831 token | 10.65 |
| Qwen3-Emb → Jev, ilk 100 | 94.81 | 96.90 | 97.62 | 97.49 | 91.33 | 41.9 | 299 | 3,151 token | 0.37 |
| Qwen3-Emb → Jev, ilk 20, documentation | 94.72 | 96.27 | 96.99 | 96.88 | 91.98 | 40.0 | 288 | 1,688 token | 0.20 |
| heads → CLM-8B, ilk 100 | 13.11 | 16.10 | 25.81 | 25.68 | 3.76 | 2.4 | — | — | — |
| heads → CLM-8B, ilk 20 | 33.22 | 39.42 | 65.25 | 64.97 | 10.10 | 1.1 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 20.01 | 23.59 | 37.67 | 37.54 | 7.23 | 2.3 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 27.64 | 32.68 | 60.30 | 60.10 | 5.27 | 1.2 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 12.76 | 15.20 | 24.79 | 24.71 | 4.08 | 6.2 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 32.60 | 38.20 | 64.40 | 64.04 | 9.63 | 26.8 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 26.65 | 35.79 | 57.03 | 56.88 | 3.94 | 1.1 | — | — | — |
| heads → Qwen3-Reranker-8B, ilk 20, documentation | 94.57 | 96.64 | 97.30 | 97.21 | 91.26 | 298.0 | 2476 | 20 çift | — |
| heads → bge-reranker-v2-gemma, ilk 20, documentation | 71.41 | 85.32 | 94.56 | 94.45 | 48.24 | 103.4 | 773 | 20 çift | — |
| Qwen3-Emb + LoRA | 93.67 | 96.53 | 97.93 | 97.89 | 88.57 | 5.4 | — | — | — |
| Qwen3-Emb + LoRA + head (epoch 0 = kimlik) | 93.67 | 96.53 | 97.93 | 97.89 | 88.57 | 4.9 | — | — | — |

### ToolRet, kategori bazında NDCG@10: heads → heads + Jev 100

| Kategori | heads ↑ % | heads + Jev 100 ↑ % | fark ↑ puan |
|---|---:|---:|---:|
| code | 54.40 | 56.60 | +2.19 |
| customized | 46.58 | 52.73 | +6.15 |
| web | 40.42 | 44.90 | +4.47 |

### ToolRet, görev bazında en büyük hareketler (NDCG@10): heads → heads + Jev 100

| Görev | n | heads ↑ % | heads + Jev 100 ↑ % | fark ↑ puan |
|---|---:|---:|---:|---:|
| autotools-food | 22 | 49.37 | 32.83 | -16.54 |
| appbench | 32 | 62.67 | 52.99 | -9.67 |
| gorilla-huggingface | 500 | 44.58 | 37.12 | -7.46 |
| toolbench | 1100 | 58.38 | 52.68 | -5.70 |
| gorilla-tensor | 55 | 33.50 | 28.74 | -4.76 |
| toollens | 314 | 11.70 | 7.85 | -3.85 |
| taskbench-huggingface | 23 | 31.49 | 45.26 | +13.78 |
| gta | 14 | 25.52 | 39.84 | +14.32 |
| gpt4tools | 32 | 76.14 | 92.35 | +16.21 |
| taskbench-multimedia | 40 | 62.85 | 79.91 | +17.06 |
| toolemu | 38 | 29.78 | 59.48 | +29.69 |
| t-eval-dialog | 50 | 33.89 | 65.91 | +32.01 |

35 görevden 24 yukarı, 11 aşağı (0,5 puandan fazla).

### ToolRet, kategori bazında NDCG@10: heads → LoRA

| Kategori | heads ↑ % | LoRA ↑ % | fark ↑ puan |
|---|---:|---:|---:|
| code | 54.40 | 61.39 | +6.99 |
| customized | 46.58 | 55.29 | +8.71 |
| web | 40.42 | 46.39 | +5.97 |

### ToolRet, görev bazında en büyük hareketler (NDCG@10): heads → LoRA

| Görev | n | heads ↑ % | LoRA ↑ % | fark ↑ puan |
|---|---:|---:|---:|---:|
| restgpt-spotify | 40 | 52.01 | 41.99 | -10.02 |
| restgpt-tmdb | 54 | 31.91 | 22.93 | -8.99 |
| autotools-music | 32 | 21.29 | 18.08 | -3.21 |
| tool-be-honest | 350 | 41.53 | 39.46 | -2.07 |
| t-eval-step | 50 | 31.50 | 29.51 | -1.99 |
| apibank | 101 | 59.43 | 57.65 | -1.78 |
| toolbench | 1100 | 58.38 | 72.97 | +14.60 |
| autotools-food | 22 | 49.37 | 65.02 | +15.65 |
| taskbench-multimedia | 40 | 62.85 | 81.42 | +18.57 |
| toolemu | 38 | 29.78 | 49.80 | +20.02 |
| taskbench-huggingface | 23 | 31.49 | 60.97 | +29.48 |
| mnms | 33 | 19.63 | 50.95 | +31.31 |

35 görevden 26 yukarı, 7 aşağı (0,5 puandan fazla).
<!-- tables:end -->

Jev'in kendisi olmadan LiveMCPBench ve MCP-Zero'da BM25'i Jev'e vermek bile heads'i geçiyor; BM25'in
recall'u yetmediği ToolRet'te ise BM25 → Jev (47.81) heads'in (54.03) altında kalıyor. Hangi metnin
gönderildiği önemli: ilk 20 aracın tam dokümantasyonu ToolRet'te ilk 100 aracın ad + açıklamasından
2.65 puan daha iyi (57.69 / 55.04), MCP-Zero'da eşit, LiveMCPBench'te 2.2 puan geride (heads'in ilk
20'sinin kaçırdığı araçlar). Jev tek başına LiveMCPBench'te 65.05 ile heads → Jev'in yanına geliyor,
ama sorgu başına 40.910 token (4 çağrı) harcıyor; heads → Jev 100'ün 7.018'i.

### CLM Jev'in yerine

Aynı sarmalayıcıyla (`--rerank clm`, `adapters/rerank.py`) Jev'in yerine CLM_v0.1-8B kondu: heads'in
ilk 100 ya da 20 aracı, Faz 0'ın en iyi CLM düzeniyle (8090'daki Qwen3-8B, `example_call` metni,
`clm` sorgu formatı) yeniden puanlanıyor; tamamı Faz 0 önbelleğinden, 12 satır 2.5 dakika, ücretsiz.
Sonuç her sette listeyi bozmak: ToolRet NDCG@10 54.03 → 15.36 (ilk 100) / 28.94 (ilk 20),
LiveMCPBench 53.95 → 11.85 / 27.65, MCP-Zero top-1 79.87 → 3.76 / 10.10. Faz 0'ın fine-tune'lu CLM
head'leri (ToolRet'te tek başına 17.68) yeniden sıralayıcı olarak 34.20'ye çıkıyor, yine heads'in
20 puan altında. Derinlik 20'nin 100'den iyi olması, CLM'in listeyi rastgeleye yakın karıştırdığını
gösterir: 20 araçlık pencerede recall@10 zaten yüksek kalır. Metin biçimi açıklama değil: aynı üç
satır Jev'in okuduğu metinle (ad + açıklama 1000 karakter, dokümantasyon 3000 karakter,
`--rerank-max-chars`) tekrarlandı ve CLM her sette 1–7 puan daha da düştü (ToolRet ilk 100: 15.36 →
13.84; LiveMCPBench ilk 20: 27.65 → 21.06), yani Faz 0'ın `example_call` seçimi CLM'in en iyi
şansıydı. Jev'in katkısı modelin "ikinci aşama"
olmasından değil, araç metnini okuyup isteğe göre karar verebilmesinden geliyor; CLM soru → cevap
ve ajan adımı çiftleriyle eğitildiği için araç açıklamaları onun dağılımının dışında (Faz 0 kapı
raporu), ve 100 adaylık kısa liste bunu değiştirmiyor. Yerel bir Jev alternatifi istenirse adres,
Faz 0'da ölçülen cross-encoder sınıfı (bge-reranker-v2-gemma, ToolRet 47.52) olur, CLM değil.

### Cross-encoder Jev'in yerine

Jev'in kazancı isteği ve adayı birlikte okumaktan geliyorsa, yerel bir cross-encoder aynı koltukta
aynı işi yapmalı. vLLM'in skor API'si (`/score`) üstünden iki model, aynı sarmalayıcı (`--rerank
cross`, `adapters/cross_encoder.py`), Jev satırlarının birebir metniyle (ilk 20, dokümantasyon 3000
karakter; istek 6000 karakterde kesilir, çünkü her çiftte yinelenir): Qwen3-Reranker-8B (8095, vLLM'in
reçetesiyle orijinal model, `<Instruct>/<Query>/<Document>` şablonu) ve bge-reranker-v2-gemma (8096,
FlagEmbedding'in `A:/B:/prompt` biçimi). Puanlar `.cache/toolrank/scores.sqlite`'ta.

| heads'in ilk 20'si + dokümantasyon (↑ %) | heads | Jev | Qwen3-Reranker-8B | bge-reranker-v2-gemma |
|---|---:|---:|---:|---:|
| ToolRet NDCG@10 | 54.03 | 57.69 | 58.05 | 53.96 |
| ToolRet cat-macro NDCG@10 | 47.13 | 52.71 | 52.93 | 50.26 |
| LiveMCPBench NDCG@10 | 53.95 | 64.03 | 62.68 | 36.19 |
| MCP-Zero Precision@1 | 79.87 | 92.34 | 91.26 | 48.24 |

- **Qwen3-Reranker-8B Jev'in yerel karşılığı.** ToolRet'te aynı düzende Jev'in 0.2–0.4 puan önünde,
  MCP setlerinde 1.1–1.4 geride; ücretsiz, kendi makinede, sorgu dışarı çıkmıyor. Gecikme, GPU'yu
  başka işlerle paylaşırken 8 eşzamanlı istekle sorgu başına 0.3–0.6 s (çağrı başına 1.7–2.5 s, 20
  çift); Jev'in 0.3 s'si ile aynı sınıf. Bellek: bf16 ağırlıklar 16 GB, 8192 token pencere.
- **bge-reranker-v2-gemma yetmiyor.** ToolRet'te heads ile başa baş (53.96; cat-macro 50.26, +3),
  MCP setlerinde kısa listeyi bozuyor, LiveMCPBench'te heads'in bile altına iniyor. Faz 0'daki
  47.52 kâğıdın kendi kurulumuyla tüm külliyat üstündeydi; burada vLLM'in `no_post_processing`
  yüklemesiyle ham "Yes" logit'i, istem biçimi FlagEmbedding'in `get_inputs`'uyla aynı. MCP
  dokümantasyonunda (JSON şema, sunucu adı) 2024'ün 2B modeli zayıf; aday olarak elendi.
- Jev'in kalan satırlarının cevabı (`qwen3emb_jev*`): Jev üstteyken fine-tune'un katkısı küçülüyor.
  ToolRet ilk 100'de zero-shot Qwen3 → Jev 54.59 / cat-macro 52.05, heads → Jev 55.04 / 51.41; ilk
  20'de 55.98 / 52.67'ye karşı 57.69 / 52.71. Head'lerin değeri kısa listenin recall'unda (Recall@20
  72.51'e karşı 69.45): 100 aday verince fark kapanıyor, 20 aday verince 1.7 puan kalıyor. LoRA'nın
  hedefi de bu: daha iyi kısa liste.
- Kalan cross-encoder satırları (ilk 100, BM25 ilk 30, LiveMCPBench'te tek başına) ve LoRA'nın kısa
  listesi üstünde Qwen3-Reranker satırları koşuyor (`data/lora/chain2.sh`, `chain3.sh`); reranker'lar
  (48 GB) ile LoRA eğitimi (20–30 GB) aynı anda belleğe sığmadığı için sıraya kondu.

### LoRA: omurganın kendisi

`toolrank finetune` omurgayı dondurup vektörlerin üstünde head eğitiyordu; `scripts/lora_train.py`
omurganın kendi ağırlıklarına LoRA adaptörleri ekliyor (derece 16, alpha 32, dropout 0.05; q, k, v, o,
gate, up, down; 36 katman; 43.6 M eğitilen parametre, modelin %0.53'ü). Veri ve metin head eğitimiyle
aynı yol: ToolRet-train çiftleri, kıyaslama sorgusuna eşit 776 istek düşürülüp 20.000 çift;
`instruct_query` istek, `documentation` araç metni. Kayıp: 16 çiftlik mikro-partide InfoNCE (τ 0.05),
partinin öteki pozitifleri negatif, kazılmış negatif yok; 2 mikro-parti birikim, lr 1e-4 kosinüs, 625
adım, istek 256 / doküman 768 token; son-token havuzlama + L2, vLLM ile aynı (parite kosinüsü 0.9999).
Her 300 adımda dev (MCP-Zero `_server`) süreç içinde puanlanıp en iyi adaptör saklandı; en iyi adaptör
ağırlıklara katılıp 8097'de `qwen3-emb-lora` olarak servis edildi ve servis, süreç içi dev skorunu
yeniden üretti (93.67'ye karşı 93.65).

| Dev, MCP-Zero (seçim seti) | adım 0 | adım 300 | adım 600 | adım 625 |
|---|---:|---:|---:|---:|
| NDCG@10 ↑ % | 87.14 | 90.68 | 93.63 | 93.65 |
| Precision@1 ↑ % | 78.08 | 83.60 | 88.54 | 88.50 |

| Tek aşama, w/ inst | ToolRet NDCG@10 ↑ % | ToolRet cat-macro ↑ % | ToolRet Recall@20 ↑ % | LiveMCPBench NDCG@10 ↑ % | LiveMCPBench Recall@10 ↑ % | MCP-Zero Precision@1 ↑ % |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Embedding-8B, zero-shot | 51.11 | 46.54 | 69.45 | 53.74 | 61.09 | 78.19 |
| + v0.1 head'leri (60 bin çift) | 54.03 | 47.13 | 72.51 | 53.95 | 61.66 | 79.87 |
| + LoRA (20 bin çift) | 58.90 | 54.36 | 75.25 | 55.74 | 63.34 | 88.57 (seçim seti) |

- **ToolRet'te tek aşamada en iyi satır.** cat-macro 54.36, Faz 0 kapısının eşiği 50'nin ve head'lerin
  47.13'ünün üstünde, StackOne v2'nin 54.4'ü düzeyinde; heads → Jev (52.71) ve heads → Qwen3-Reranker
  (52.93) satırlarını ikinci aşama olmadan geçiyor. Üç kategoride de artış (code 54.40 → 61.39,
  customized 46.58 → 55.29, web 40.42 → 46.39); head'lere göre 35 görevden 26'sı yukarı, 7'si aşağı.
- **Kısa liste de iyileşti**: Recall@20 72.51 → 75.25; üstüne reranker satırları koşuyor.
- **Genelleme ılımlı.** Hiç görülmemiş LiveMCPBench'te +2.0 NDCG@10 (94 sorgu, gürültü sınırında);
  MCP-Zero adaptörün seçildiği set, +10.4 top-1 iyimser. ToolRet kazancı eğitim verisiyle aynı
  aileden; head'ler ve StackOne v2 de aynı veriyle eğitildi, karşılaştırma bu anlamda adil.
- **Head'ler LoRA'nın üstüne bir şey katmıyor.** Aynı v0.1 reçetesiyle (60 bin çift, parti içi
  negatif, lr 1e-5) LoRA omurgasında head eğitimi: dev 93.67'den hiçbir epoch'ta yukarı çıkmadı
  (93.60, 93.62, 93.17, 93.32, 93.44), seçilen epoch 0, yani kimlik; `lora_heads` satırları dense
  satırlarla aynı.
- **Maliyet**: eğitim 8.6 saat (adım başına 0.79 dk, dev ölçümü 11 dk), adaptör 167 MB, birleştirilmiş
  ağırlıklar 16 GB; ToolRet'in 44 bin aracının yeni omurgayla soğuk kodlaması 24 dakika. Çalışma
  anında fark yok: aynı model boyutu, aynı gecikme.

### Duman testi: ilk 50 sorgu

Tam koşu öncesi, aynı 50 sorguda taban satırlarıyla (`results/jevsmoke_*.json`; ToolRet'te tek
görev, craft-math-algebra): heads → Jev 100 LiveMCPBench'te 52.06 → 67.24 NDCG@10, MCP-Zero'da
top-1 92 → 94, Jev tek başına MCP-Zero'da top-1 98. 11 satır, 8.4 M token, 0.35 $.

## Komutlar

```bash
# Mac: anahtar .env'de (TYPESAFE_API_KEY=...), GB10'a tek satır olarak geçirildi (mod 600, eşlenmez)
grep '^TYPESAFE_API_KEY=' ~/toolrank/.env | ssh gb10 'umask 077; cat > ~/toolrank/.env.typesafe; chmod 600 ~/toolrank/.env.typesafe'
# GB10, ~/toolrank, 8091 ayakta, gömme önbelleği sıcak (her satırda encoder tokens 0)
set -a; . ./.env.typesafe; set +a
LIMIT=50 TAG=jevsmoke bash scripts/jev_compare.sh          # duman testi
PYTHONUNBUFFERED=1 nohup bash scripts/jev_compare.sh > data/logs/jev_compare.log 2>&1 &   # tam koşu, 25 dk (402'ye kadar)
# satırlar (scripts/jev_compare.sh): EMB = --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192
#   --emb-batch 128 --tool-format documentation --query-format instruct_query --with-inst; JEV = --jev-model jev-1.13.0 --jev-workers 8
toolrank eval --data data/$d --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz $EMB --rerank jev --rerank-depth 100 $JEV
toolrank eval --data data/$d --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz $EMB --rerank jev --rerank-depth 20 --jev-tool-format documentation --jev-max-chars 3000 $JEV
toolrank eval --data data/$d --scorer bm25 --no-stem --tool-format documentation --with-inst --rerank jev --rerank-depth 30 $JEV
toolrank eval --data data/$d --scorer jev --with-inst --tool-format name_desc --jev-chunk 200 --jev-per-chunk 20 $JEV   # MCP setleri
# mcp_zero_server için --ks 1,5,10,20; taban satırları (ilk 50): aynı bayraklar, --rerank yok, --out results/jevsmoke_<set>_base_<row>.json
# CLM in Jev's role (GB10, 8090 + 8091 up, free): scripts/clm_rerank.sh, i.e. per set
toolrank eval --data data/$d --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz $EMB --rerank clm \
  --rerank-emb-url http://127.0.0.1:8090/v1 --rerank-emb-model qwen3-8b --rerank-truncate 2048 \
  --rerank-tool-format example_call --rerank-query-format clm --rerank-clm-ckpt ~/.cache/clm/CLM_v0.1-8B.pt --rerank-depth 100
# ... --rerank-depth 20; --rerank-clm-ckpt data/heads/clm_60k_lr1e-2.pt; BM25 base with --rerank-depth 30
PYTHONUNBUFFERED=1 nohup bash scripts/clm_rerank.sh > data/logs/clm_rerank.log 2>&1 &
# the same with exactly Jev's text (results/clmj_*.json): name_desc cut to 1000, the top 20 with documentation cut to 3000
CLM_FORMAT=name_desc CLM_MAX_CHARS=1000 ROWS="heads_clm100 heads_clm20doc bm25_clm30" TAG=clmj bash scripts/clm_rerank.sh
# cross-encoders (GB10): the rerankers as vLLM sequence classifiers, then the same texts as the Jev rows
docker compose -f deploy/spark/compose.yaml --profile rerank up -d qwen3-reranker bge-reranker   # 8095 / 8096
ROWS=heads_x20doc PYTHONUNBUFFERED=1 nohup bash scripts/cross_rerank.sh > data/logs/cross_rerank.log 2>&1 &   # phase A
#   i.e. toolrank eval ... --rerank cross --rerank-emb-url http://127.0.0.1:8095 --rerank-emb-model qwen3-reranker \
#     --rerank-template qwen3 --rerank-depth 20 --rerank-tool-format documentation --rerank-max-chars 3000 --rerank-workers 8
ROWS="heads_x100 bm25_x30 alone" bash scripts/cross_rerank.sh                                     # phase B, after LoRA
# LoRA (GB10, [lora] extra; the smoke run first: --n-train 64 --micro-batch 4 --eval-every 8 --limit-dev 200 --no-merge)
uv run --extra lora python scripts/lora_train.py --pairs data/toolret_train/pairs.jsonl --dev data/mcp_zero_server \
  --eval data/toolret --eval data/livemcpbench_server --n-train 20000 --micro-batch 16 --accumulate 2 \
  --eval-every 300 --check-parity --out data/lora/qwen3-emb-lora-20k
TOOLRANK_LORA=$HOME/toolrank/data/lora/qwen3-emb-lora-20k/merged docker compose -f deploy/spark/compose.yaml --profile lora up -d qwen3-emb-lora  # 8097
toolrank eval --data data/$d --scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 \
  --tool-format documentation --query-format instruct_query --with-inst --out results/lora_${d}_lora.json
toolrank finetune --data data/toolret_train/pairs.jsonl --n-train 60000 --n-val 2000 --dev data/mcp_zero_server \
  --eval data/toolret --eval data/livemcpbench_server --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora \
  --out data/heads/lora_60k.pt --npz dist/heads/lora_60k.npz        # then --scorer clm --clm-ckpt dist/heads/lora_60k.npz rows
# Mac
scp 'gb10:toolrank/results/jev*_*.json' 'gb10:toolrank/results/clm*_*.json' 'gb10:toolrank/results/cross*_*.json' 'gb10:toolrank/results/lora_*.json' results/
uv run python scripts/rerank_report.py --write      # the summary and the tables of this report, from the result files
toolrank compare results/jev_*.json results/clm_*.json results/cross_*.json docs/results/readme_*.json --metrics NDCG@10,Recall@5,Recall@10,Precision@1
```

## Ortam

GB10 (NVIDIA GB10, Ubuntu 24.04 aarch64), vLLM NGC `nvcr.io/nvidia/vllm:26.01-py3`, Qwen3-Embedding-8B
bf16 `--max-model-len 8192` (8091), paketlenmiş head'ler v0.1 (`.npz`, numpy). toolrank 0.2.0.dev0,
commit `c40419e`. Jev: `api.typesafe.ai`, `jev-1.13.0`, 8 eşzamanlı istek, çağrılar GB10'dan
(İstanbul; TypeSafe'in sunucuları Batı Kıyısı'nda, Mac'ten tek küçük istek 0.30 s, GB10'dan p50
285–360 ms, p95 415–540 ms). Gömme önbelleği sıcak; Jev önbelleği duman testinin 50 sorgusu dışında
soğuk. Harcama: 17 Jev satırı toplam 428.5 M token = 18.00 $ (ilk 10 satır 4.66 $, zero-shot → Jev
satırları 3.8 $, MCP-Zero tek başına 9.51 $), duman testi 0.35 $. İlk koşuda hesabın kredisi tek
başına satırında bitmiş (HTTP 402), 15 $ eklenince kalan satırlar önbellekten devam etti.
CLM satırları: Qwen3-8B pooling (8090, `--max-model-len 2048`), `~/.cache/clm/CLM_v0.1-8B.pt` ve
`data/heads/clm_60k_lr1e-2.pt`, head'ler torch ile GPU'da, gömmeler Faz 0 matrisinin önbelleğinden
(encoder tokens 0); commit `c4723b1`, birebir metin satırları `76472ec` (3000 karakterde kesilen uzun
dokümantasyonlar 8090'da yeniden kodlandı).
Cross-encoder satırları: aynı NGC imajı, Qwen3-Reranker-8B (8095, bellek payı 0.25) ve
bge-reranker-v2-gemma (8096, 0.15), ikisi de 8192 token pencere, vLLM'in sıralama sınıflandırıcısı
yüklemesi (`hf_overrides`), 8 eşzamanlı istek, puanlar `scores.sqlite`'ta; GPU'yu LoRA koşusu ve 8091
ile paylaşırken; commit'ler `645b026` … `4569341`. LoRA: `scripts/lora_train.py` (`[lora]` ekstrası: transformers 5.18, peft), GB10'da 10:01–18:54,
516.8 dakika eğitim + birleştirme; çıktılar `data/lora/qwen3-emb-lora-20k/{adapter,merged,train.json}`;
servis `--profile lora`, 8097, `qwen3-emb-lora`; LoRA satırları ve head eğitimi `data/lora/chain2.sh`,
LoRA üstünde reranker `chain3.sh`. Commit'ler `645b026` (betik), `8000746` (birleştirilmiş klasör
düzeltmesi).

## Sapmalar ve açıklamalar

- ToolRet'te mikro ortalama cat-macro kadar artmıyor (heads → Jev 100: +1.0 / +4.3): 35 görevden 24
  yukarı, 11 aşağı; düşenler büyük görevler, toolbench (n 1100) −5.7 ve gorilla-huggingface (n 500)
  −7.5, çıkanlar küçük görevler, t-eval-dialog +32.0, toolemu +29.7, taskbench-multimedia +17.1.
  Büyük görevlerde aynı aileden çok sayıda benzer araç var; ad + açıklama 1000 karaktere kesilince
  ayırt edici parça düşüyor olabilir: tam dokümantasyonla ilk 20 (57.69) bunun bir kısmını geri
  alıyor. Derinlik ve metin taraması (30 / 50 araç, 2000 karakter) yapılmadı.
- MCP-Zero'nun istekleri her aracın açıklamasından Qwen3-8B tarafından yazıldı; açıklamayı okuyan
  bir modelin burada güçlü olması beklenir (BM25 → Jev 30 bile top-1 90.29). ToolRet'teki artış
  daha gerçekçi bir ölçü.
- Jev'in çağrı başına p50'si 285–360 ms; heads'in sorgu başına 0.86 ms'i (sıcak) yanında serve'deki
  `search_tools` p50'sini 5 ms'den 300 ms'in üstüne çıkarır. Sorgu metni dış bir servise gider;
  kullanım günlüğündeki HMAC gizliliği bu yolda geçerli değil.
- Jev tek başına MCP-Zero'da (14 parça + son tur, sorgu başına 15 çağrı) heads → Jev'in 1.5 puan
  altında: parçalı sıralama, kazananları tek listede karşılaştıran ikinci tura rağmen, iyi bir kısa
  listeden daha kötü. Gömme modeli ucuz ve iyi bir ilk aşama olarak kalıyor.
- Duman testinin ToolRet satırları (ilk 50 sorgu, tek görev) tavana yakındı ve yanıltıcıydı; tam
  koşu şart.
- CLM satırlarında ikinci encoder'ın önbellek ıskaları rapora yazılmıyor (`encoder_tokens` taban
  scorer'ın); hepsi önbellekten geldiği için burada fark etmedi.
- Birleştirilmiş LoRA ağırlıkları ilk seferde servis edilemedi: eğitim ortamındaki transformers 5.18,
  `tokenizer_config.json`'da `extra_special_tokens`'ı liste olarak yazdı (NGC imajındaki 4.51 bunu
  sözlük bekleyip çöküyor) ve `config.json`'da `rope_theta` yerine `rope_parameters` yazdı (4.51 onu
  görmeyip varsayılan theta ile yanlış vektör üretirdi). Birleştirilmiş klasöre orijinal deponun
  config, tokenizer ve havuzlama dosyaları kondu; betik artık bunu kendisi yapıyor. Zincir bu yüzden
  18:55–22:35 arasında bekledi. Doğrulama: servis MCP-Zero'da 93.67 / 88.57, süreç içi 93.65 / 88.50.
- LoRA eğitiminde dokümanlar 768, istekler 256 token'da kesildi; serviste pencere 8192. Uzun
  dokümantasyonlu araçlarda eğitim ve servis metni aynı değil; etkisi ölçülmedi.

## Sözleşme notu

Master Customer Agreement 2.3(b): Servis veya çıktıları model damıtma, çıktıyı taklit eden model
eğitimi ya da benzer/rakip ürün geliştirmeyi kolaylaştırmak için kullanılamaz. Burada yapılan, içeride
ölçüm ve isteğe bağlı bir adapter; Jev'in sıralamaları hiçbir eğitim sinyaline girmiyor ve girmemeli.
Sonuçları Jev adıyla yayımlamadan önce TypeSafe'e sorulmalı (madde 16: karşı tarafın adını kullanma
hakkı yok). README tablosuna Jev satırı konmadı.

## Sonraki

- Koşuyor: cross-encoder'ın kalan satırları (ilk 100, BM25 ilk 30, tek başına) ve LoRA'nın kısa listesi
  üstünde Qwen3-Reranker-8B; gelince `rerank_report.py --write`.
- LoRA'yı büyütmek: 60 bin ve 206 bin çift, daha uzun doküman penceresi (1024–2048 token), ikinci epoch;
  her biri 8–24 saatlik GB10 koşusu. Head'lerde daha çok veri yalnız büyük görevlere yaramıştı; LoRA'da
  ölçülmedi.
- Ürün kararı 1, omurga: paketlenen varsayılanın LoRA'lı omurga olması (16 GB birleştirilmiş ağırlık ya
  da 167 MB adaptör; Qwen3-Embedding-8B Apache 2.0, eğitim verisinin lisanssızlığı head'lerdeki gibi
  model kartına yazılır). README tablosu ve leaderboard satırı buna göre yenilenir; FP8 ile uyumu
  ölçülmeli.
- Ürün kararı 2, ikinci aşama: `search` / `serve` için isteğe bağlı reranker; yerelde Qwen3-Reranker-8B
  (+16 GB, +0.3–0.6 s), barındırılan için Jev (+0.3 s, sorgu dışarı çıkar). MCP setlerindeki +9–12 puan
  bunu değerli kılıyor; ToolRet'te LoRA tek başına yetiyor.
- Bir derinlik / metin taraması (ilk 30 ve 50, documentation 2000 karakter) ve uyarlanabilir K için
  ikinci aşama skorlarıyla kesim (`score_kind` bugün Jev ve cross satırlarında cut'ı reddediyor).
- Yayın için TypeSafe'e sormak (Jev satırları); sözleşme gereği Jev çıktıları eğitim sinyali olamaz.
