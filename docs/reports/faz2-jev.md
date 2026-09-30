# Faz 2 — Jev karşılaştırması (30 Eyl – 1 Eki 2026)

TypeSafe AI'ın 15 Eylül 2026'da duyurduğu "System One" modeli Jev, toolrank'in kendi scorer'larının
yanında, aynı protokol, aynı setler ve aynı araç metniyle ölçüldü. Plan maddesi değil; Faz 2'nin
1. haftasında, `toolrank eval --rerank jev` ve `--scorer jev` ile (`adapters/jev.py`).

## Sonuç (tek cümle)

toolrank'in kısa listesinin üstünde ikinci aşama olarak Jev üç sette de kazandırıyor: ToolRet'te
heads'in ilk 20'sini tam dokümantasyonla yeniden sıralamak NDCG@10'u 54.03'ten 57.69'a ve cat-macro'yu
47.13'ten 52.71'e (Faz 0 kapısının 50 eşiği, ama dış bir API ile), LiveMCPBench'te ilk 100 ile
NDCG@10'u 53.95'ten 66.25'e, MCP-Zero'da top-1'i 79.87'den 91.55'e taşıyor; bedeli sorgu başına
0.3 s ve 0.0002 $. Jev tek başına LiveMCPBench'te aynı yere 6 kat token ile geliyor; 255 seçenek
sınırı yüzünden ToolRet'te tek başına koşamıyor, MCP-Zero'daki tek başına satırı kredi bitince
(HTTP 402) tamamlanamadı.

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

| Kol | ToolRet | LiveMCPBench | MCP-Zero |
|---|---|---|---|
| BM25 (w/ inst), Qwen3-Embedding-8B, heads | README satırları | README satırları | README satırları |
| heads ilk 100 → Jev, name_desc, 1000 karakter | evet | evet | evet |
| heads ilk 20 → Jev, documentation, 3000 karakter | evet | evet | evet |
| BM25 ilk 30 → Jev, name_desc (cookbook düzeni) | evet | evet | evet |
| Jev tek başına, parça 200, parça başı 20 kazanan | hayır (44k araç) | evet (3 parça) | 402, eksik |

## Tablo

w/ inst; taban satırları `docs/results/readme_*.json` (README tablosu), Jev satırları
`results/jev_*.json`. Son üç sütun Jev'e ait: GB10'dan çağrı başına p50, sorgu başına faturalanan
token ve satırın toplam ücreti. Ücretler 0.042 $ / M token ile.

### ToolRet (w/ inst, n=7961)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | cat-macro | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 39.27 | 42.42 | 49.49 | 40.70 | 36.41 | — | — | — |
| BM25 → Jev, ilk 30 | 47.81 | 51.56 | 56.30 | 46.72 | 45.19 | 285 | 1,965 | 0.66 $ |
| Qwen3-Embedding-8B | 51.11 | 54.24 | 62.32 | 51.60 | 46.54 | — | — | — |
| heads | 54.03 | 57.39 | 65.59 | 55.06 | 47.13 | — | — | — |
| heads → Jev, ilk 100 | 55.04 | 58.25 | 66.76 | 56.81 | 51.41 | 310 | 5,564 | 1.86 $ |
| heads → Jev, ilk 20, documentation | 57.69 | 61.56 | 68.35 | 58.30 | 52.71 | 303 | 4,047 | 1.35 $ |

### LiveMCPBench (w/ inst, n=94)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 25.38 | 22.92 | 31.41 | 17.02 | — | — | — |
| BM25 → Jev, ilk 30 | 39.47 | 38.09 | 44.07 | 22.34 | 298 | 3,438 | 0.01 $ |
| Qwen3-Embedding-8B | 53.74 | 50.82 | 61.09 | 37.23 | — | — | — |
| heads | 53.95 | 53.03 | 61.66 | 36.17 | — | — | — |
| heads → Jev, ilk 100 | 66.25 | 65.84 | 73.68 | 46.81 | 329 | 7,018 | 0.03 $ |
| heads → Jev, ilk 20, documentation | 64.03 | 64.70 | 69.63 | 44.68 | 313 | 4,984 | 0.02 $ |
| Jev tek başına | 65.05 | 66.44 | 73.81 | 45.74 | 360 | 40,910 | 0.16 $ |

### MCP-Zero (w/ inst, n=2792)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | Precision@1 | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 63.32 | 73.94 | 84.95 | 84.78 | 45.63 | — | — | — |
| BM25 → Jev, ilk 30 | 92.84 | 94.55 | 94.89 | 94.70 | 90.29 | 286 | 1,289 | 0.15 $ |
| Qwen3-Embedding-8B | 87.21 | 92.31 | 95.63 | 95.52 | 78.19 | — | — | — |
| heads | 88.53 | 94.20 | 96.12 | 96.02 | 79.87 | — | — | — |
| heads → Jev, ilk 100 | 94.90 | 96.96 | 97.55 | 97.46 | 91.55 | 292 | 3,104 | 0.36 $ |
| heads → Jev, ilk 20, documentation | 95.01 | 96.55 | 97.23 | 97.13 | 92.34 | 285 | 1,796 | 0.21 $ |

### ToolRet, kategori bazında NDCG@10: heads → heads + Jev 100

| Kategori | heads | heads → Jev | fark |
|---|---:|---:|---:|
| code | 54.40 | 56.60 | +2.19 |
| customized | 46.58 | 52.73 | +6.15 |
| web | 40.42 | 44.90 | +4.47 |

### ToolRet, görev bazında en büyük hareketler (NDCG@10)

| Görev | n | heads | heads → Jev | fark |
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

Jev'in kendisi olmadan LiveMCPBench ve MCP-Zero'da BM25'i Jev'e vermek bile heads'i geçiyor; BM25'in
recall'u yetmediği ToolRet'te ise BM25 → Jev (47.81) heads'in (54.03) altında kalıyor. Hangi metnin
gönderildiği önemli: ilk 20 aracın tam dokümantasyonu ToolRet'te ilk 100 aracın ad + açıklamasından
2.65 puan daha iyi (57.69 / 55.04), MCP-Zero'da eşit, LiveMCPBench'te 2.2 puan geride (heads'in ilk
20'sinin kaçırdığı araçlar). Jev tek başına LiveMCPBench'te 65.05 ile heads → Jev'in yanına geliyor,
ama sorgu başına 40.910 token (4 çağrı) harcıyor; heads → Jev 100'ün 7.018'i.

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
# Mac
scp 'gb10:toolrank/results/jev*_*.json' results/
toolrank compare results/jev_*.json docs/results/readme_*.json --metrics NDCG@10,Recall@5,Recall@10,Precision@1
```

## Ortam

GB10 (NVIDIA GB10, Ubuntu 24.04 aarch64), vLLM NGC `nvcr.io/nvidia/vllm:26.01-py3`, Qwen3-Embedding-8B
bf16 `--max-model-len 8192` (8091), paketlenmiş head'ler v0.1 (`.npz`, numpy). toolrank 0.2.0.dev0,
commit `c40419e`. Jev: `api.typesafe.ai`, `jev-1.13.0`, 8 eşzamanlı istek, çağrılar GB10'dan
(İstanbul; TypeSafe'in sunucuları Batı Kıyısı'nda, Mac'ten tek küçük istek 0.30 s, GB10'dan p50
285–360 ms, p95 415–540 ms). Gömme önbelleği sıcak; Jev önbelleği duman testinin 50 sorgusu dışında
soğuk. Harcama: tam koşunun 10 satırı 111.0 M token = 4.66 $, duman testi 0.35 $; hesabın kredisi
MCP-Zero'nun tek başına satırında (~3.240 çağrıdan sonra, tahmini 247 M token = 10.4 $ daha) bitti.

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
- Jev tek başına MCP-Zero satırı yok: hesap kredisi bitti (HTTP 402 `billing_error`). Kredi
  eklenirse `bash scripts/jev_compare.sh` kaldığı yerden devam eder; yapılan ~3.240 çağrı önbellekte.
- Duman testinin ToolRet satırları (ilk 50 sorgu, tek görev) tavana yakındı ve yanıltıcıydı; tam
  koşu şart.

## Sözleşme notu

Master Customer Agreement 2.3(b): Servis veya çıktıları model damıtma, çıktıyı taklit eden model
eğitimi ya da benzer/rakip ürün geliştirmeyi kolaylaştırmak için kullanılamaz. Burada yapılan, içeride
ölçüm ve isteğe bağlı bir adapter; Jev'in sıralamaları hiçbir eğitim sinyaline girmiyor ve girmemeli.
Sonuçları Jev adıyla yayımlamadan önce TypeSafe'e sorulmalı (madde 16: karşı tarafın adını kullanma
hakkı yok). README tablosuna Jev satırı konmadı.

## Sonraki

- Karar: `toolrank search` / `serve` için isteğe bağlı `--rerank jev` (bugün yalnız eval'de). Bedeli
  gecikme (≈ +300 ms) ve sorgu metninin dışarı çıkması; kazancı üç sette de ölçüldü.
- Kredi eklenirse MCP-Zero tek başına satırı (~10 $) ve bir derinlik / metin taraması (ilk 30 ve 50,
  documentation 2000 karakter) ile uyarlanabilir K için Noul kapısı (cut.AdaptiveK'nın kosinüs marjı
  Jev olasılıklarına uymuyor; `score_kind = "jev"` bugün cut'ı reddediyor).
- Yayın için TypeSafe'e sormak.
