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
(HTTP 402) tamamlanamadı. Yerel alternatif olarak CLM_v0.1-8B aynı role konduğunda listeyi bozuyor:
heads'in ilk 100'ü ToolRet'te 54.03'ten 15.36'ya (Faz 0'ın fine-tune'lu CLM'iyle 34.20'ye) düşüyor;
araç seçimi CLM'in eğitim dağılımının dışında ve kısa liste bunu değiştirmiyor.

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
| heads ilk 100 ve ilk 20 → CLM_v0.1-8B (8090, example_call, `clm` sorgu formatı) | evet | evet | evet |
| heads ilk 100 → fine-tune'lu CLM (`clm_60k_lr1e-2.pt`) | evet | evet | evet |
| BM25 ilk 30 → CLM_v0.1-8B | evet | evet | evet |
| aynı üç satır, Jev'in okuduğu metinle: name_desc 1000 karakter, documentation 3000 karakter (`--rerank-max-chars`) | evet | evet | evet |

## Tablo

w/ inst; taban satırları `docs/results/readme_*.json` (README tablosu), Jev satırları
`results/jev_*.json`, CLM satırları `results/clm_*.json` (`scripts/clm_rerank.sh`, `--rerank clm`).
"sorgu p50 ms" eval'in `rank` adımı: Jev satırlarında 8 eşzamanlı isteğin toplam süresi bölü sorgu
sayısı, yani verim, çağrı başı gecikme yandaki sütunda; CLM satırlarında GPU'daki head'ler ve
önbellek aramaları. Son üç sütun Jev'e ait: GB10'dan çağrı başına p50, sorgu başına faturalanan token
ve satırın toplam ücreti, 0.042 $ / M token ile.

### ToolRet (w/ inst, n=7961)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | cat-macro | sorgu p50 ms | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 39.27 | 42.42 | 49.49 | 40.70 | 36.41 | 0.4 | — | — | — |
| BM25 → Jev, ilk 30 | 47.81 | 51.56 | 56.30 | 46.72 | 45.19 | 39.8 | 285 | 1,965 | 0.66 $ |
| Qwen3-Embedding-8B | 51.11 | 54.24 | 62.32 | 51.60 | 46.54 | 0.8 | — | — | — |
| heads | 54.03 | 57.39 | 65.59 | 55.06 | 47.13 | 0.9 | — | — | — |
| heads → Jev, ilk 100 | 55.04 | 58.25 | 66.76 | 56.81 | 51.41 | 43.1 | 310 | 5,564 | 1.86 $ |
| heads → Jev, ilk 20, documentation | 57.69 | 61.56 | 68.35 | 58.30 | 52.71 | 42.0 | 303 | 4,047 | 1.35 $ |
| heads → CLM-8B, ilk 100 | 15.36 | 17.25 | 25.56 | 20.31 | 12.47 | 3.6 | — | — | — |
| heads → CLM-8B, ilk 20 | 28.94 | 31.37 | 48.72 | 38.61 | 24.80 | 1.9 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 34.20 | 36.50 | 46.57 | 37.77 | 22.68 | 3.2 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 22.33 | 25.39 | 36.33 | 29.83 | 19.59 | 1.7 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 13.84 | 15.26 | 23.61 | 18.62 | 11.03 | 3.8 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 28.65 | 31.21 | 48.09 | 38.49 | 23.89 | 7.0 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 21.08 | 23.63 | 34.86 | 28.64 | 18.56 | 1.8 | — | — | — |

### LiveMCPBench (w/ inst, n=94)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | sorgu p50 ms | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 25.38 | 22.92 | 31.41 | 17.02 | 0.3 | — | — | — |
| BM25 → Jev, ilk 30 | 39.47 | 38.09 | 44.07 | 22.34 | 11.0 | 298 | 3,438 | 0.01 $ |
| Qwen3-Embedding-8B | 53.74 | 50.82 | 61.09 | 37.23 | 0.4 | — | — | — |
| heads | 53.95 | 53.03 | 61.66 | 36.17 | 0.1 | — | — | — |
| heads → Jev, ilk 100 | 66.25 | 65.84 | 73.68 | 46.81 | 12.2 | 329 | 7,018 | 0.03 $ |
| heads → Jev, ilk 20, documentation | 64.03 | 64.70 | 69.63 | 44.68 | 11.7 | 313 | 4,984 | 0.02 $ |
| Jev tek başına | 65.05 | 66.44 | 73.81 | 45.74 | 49.5 | 360 | 40,910 | 0.16 $ |
| heads → CLM-8B, ilk 100 | 11.85 | 12.36 | 19.79 | 8.51 | 5.5 | — | — | — |
| heads → CLM-8B, ilk 20 | 27.65 | 29.27 | 43.90 | 15.96 | 3.9 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 13.49 | 11.23 | 21.22 | 10.64 | 5.0 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 15.07 | 15.47 | 22.91 | 9.57 | 4.0 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 10.93 | 10.76 | 18.46 | 9.57 | 135.0 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 21.06 | 16.98 | 36.90 | 14.89 | 323.1 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 13.48 | 12.75 | 22.65 | 9.57 | 4.1 | — | — | — |

### MCP-Zero (w/ inst, n=2792)

| Satır | NDCG@10 | Recall@5 | Recall@10 | Comprehensiveness@10 | Precision@1 | sorgu p50 ms | Jev çağrı p50 ms | token / sorgu | ücret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 (w/ inst) | 63.32 | 73.94 | 84.95 | 84.78 | 45.63 | 0.1 | — | — | — |
| BM25 → Jev, ilk 30 | 92.84 | 94.55 | 94.89 | 94.70 | 90.29 | 39.1 | 286 | 1,289 | 0.15 $ |
| Qwen3-Embedding-8B | 87.21 | 92.31 | 95.63 | 95.52 | 78.19 | 0.1 | — | — | — |
| heads | 88.53 | 94.20 | 96.12 | 96.02 | 79.87 | 0.1 | — | — | — |
| heads → Jev, ilk 100 | 94.90 | 96.96 | 97.55 | 97.46 | 91.55 | 41.0 | 292 | 3,104 | 0.36 $ |
| heads → Jev, ilk 20, documentation | 95.01 | 96.55 | 97.23 | 97.13 | 92.34 | 38.8 | 285 | 1,796 | 0.21 $ |
| heads → CLM-8B, ilk 100 | 13.11 | 16.10 | 25.81 | 25.68 | 3.76 | 2.4 | — | — | — |
| heads → CLM-8B, ilk 20 | 33.22 | 39.42 | 65.25 | 64.97 | 10.10 | 1.1 | — | — | — |
| heads → CLM fine-tune, ilk 100 | 20.01 | 23.59 | 37.67 | 37.54 | 7.23 | 2.3 | — | — | — |
| BM25 → CLM-8B, ilk 30 | 27.64 | 32.68 | 60.30 | 60.10 | 5.27 | 1.2 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 100 | 12.76 | 15.20 | 24.79 | 24.71 | 4.08 | 6.2 | — | — | — |
| heads → CLM-8B, Jev'in metni, ilk 20, documentation | 32.60 | 38.20 | 64.40 | 64.04 | 9.63 | 26.8 | — | — | — |
| BM25 → CLM-8B, Jev'in metni, ilk 30 | 26.65 | 35.79 | 57.03 | 56.88 | 3.94 | 1.1 | — | — | — |

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
# Mac
scp 'gb10:toolrank/results/jev*_*.json' 'gb10:toolrank/results/clm_*.json' results/
toolrank compare results/jev_*.json results/clm_*.json docs/results/readme_*.json --metrics NDCG@10,Recall@5,Recall@10,Precision@1
```

## Ortam

GB10 (NVIDIA GB10, Ubuntu 24.04 aarch64), vLLM NGC `nvcr.io/nvidia/vllm:26.01-py3`, Qwen3-Embedding-8B
bf16 `--max-model-len 8192` (8091), paketlenmiş head'ler v0.1 (`.npz`, numpy). toolrank 0.2.0.dev0,
commit `c40419e`. Jev: `api.typesafe.ai`, `jev-1.13.0`, 8 eşzamanlı istek, çağrılar GB10'dan
(İstanbul; TypeSafe'in sunucuları Batı Kıyısı'nda, Mac'ten tek küçük istek 0.30 s, GB10'dan p50
285–360 ms, p95 415–540 ms). Gömme önbelleği sıcak; Jev önbelleği duman testinin 50 sorgusu dışında
soğuk. Harcama: tam koşunun 10 satırı 111.0 M token = 4.66 $, duman testi 0.35 $; hesabın kredisi
MCP-Zero'nun tek başına satırında (~3.240 çağrıdan sonra, tahmini 247 M token = 10.4 $ daha) bitti.
CLM satırları: Qwen3-8B pooling (8090, `--max-model-len 2048`), `~/.cache/clm/CLM_v0.1-8B.pt` ve
`data/heads/clm_60k_lr1e-2.pt`, head'ler torch ile GPU'da, gömmeler Faz 0 matrisinin önbelleğinden
(encoder tokens 0); commit `c4723b1`, birebir metin satırları `76472ec` (3000 karakterde kesilen uzun
dokümantasyonlar 8090'da yeniden kodlandı).

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
- CLM satırlarında ikinci encoder'ın önbellek ıskaları rapora yazılmıyor (`encoder_tokens` taban
  scorer'ın); hepsi önbellekten geldiği için burada fark etmedi.

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
- Yerel bir ikinci aşama isteniyorsa CLM değil cross-encoder: Faz 0'daki bge-reranker-v2-gemma
  düzenini `--rerank` arayüzüne bir adapter olarak takmak (`score_tools` yeterli) ve aynı 12 satırı
  koşmak.
