# Faz 1 — Hafta 2 raporu (30 Eyl 2026)

Takvimde 9–13 Kasım; erken yapıldı.

## Sonuç (tek cümle)

Ingest dizini artık aranabilir ve dağıtılabilir. `toolrank search`, kalıcı bir indeksle yalnız yeni ya
da değişen tool'ları gömüyor. Uyarlanabilir K, aynı ortalama tool sayısında üç sette de sabit top-k'dan
iyi. Kazanan head torch'suz 60 MB'lık bir `.npz` olarak paketlendi ve torch ile aynı sonuçları veriyor.
FAISS HNSW 44 bin tool'da tek sorguyu 1.6 ms'de (exact 8.2 ms) neredeyse kayıpsız cevaplıyor. Hibrit
(BM25 + RRF) yalnız MCP-Zero'da kazandırdı; varsayılan kapalı.

## 1. Index port'u ve action-vector cache

- `VectorIndex` port'u (`hashes`, tek atomik `apply`, `search`) ve üç adapter var: numpy
  (varsayılan, exact), FAISS HNSW, pgvector.
- Her satır `sha256(fingerprint ‖ tool metni)` ile anahtarlanıyor. Fingerprint endpoint, model,
  kırpma, tool formatı ve head dosyasını kapsıyor. Kalıcı bir indekste `index()` yalnız yeni ya da
  değişen tool'ları gömüp projekte ediyor, kaybolanları siliyor.
- Numpy indeksi tek bir `.npz` snapshot'ı, `flock` altında atomik olarak yazılıyor.
- Eşlik: yeni kodla Faz 0'ın tüm veriyle eğitilen head koşusu ToolRet'te tekrarlandı. Bütün genel,
  görev bazlı ve cat-macro değerler ile scorer adı birebir aynı çıktı.
- Scorer kurulumu `toolrank.build`'e taşındı; eval, search ve Hafta 3'ün sunucusu aynı kodu kullanıyor.

İndeks adapter'ları (tüm veriyle eğitilen head'ler, w/ inst):

| Run | dataset | n | NDCG@10 | Recall@10 | Comprehensiveness@10 | NDCG@10 cat-macro | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy (batch 64) | toolret | 7961 | 54.03 | 65.59 | 55.06 | 47.14 | 0.809 |
| faiss-hnsw (batch 64) | toolret | 7961 | 53.95 | 65.45 | 54.91 | 47.05 | 0.376 |
| numpy (batch 1) | toolret | 7961 | 54.04 | 65.59 | 55.06 | 47.15 | 8.209 |
| faiss-hnsw (batch 1) | toolret | 7961 | 53.95 | 65.45 | 54.91 | 47.05 | 1.572 |

- HNSW (M 32, efConstruction 200, efSearch 128) 44.453 tool'da 30 sn'de kuruluyor. NDCG@10'da 0.08,
  cat-macro'da 0.09 kaybettiriyor; tek sorguyu exact taramadan 5 kat hızlı cevaplıyor.
- Numpy'nin batch 1'de 54.04 vermesi, sorgu projeksiyonunun batch boyuna bağlı float farkı; eşitlik
  düzeyinde.
- pgvector (Postgres 17, exact tarama) LiveMCPBench ve MCP-Zero'da numpy ile birebir aynı metrikleri
  veriyor. Sorgu başına 4.8 ms (525 tool) ve 17 ms (2.792 tool) sürüyor.
- pgvector'ün HNSW'si 2.000 (`vector`) / 4.000 (`halfvec`) boyutta duruyor. 4096 boyut için ANN
  (binary quantization ya da subvector + yeniden sıralama) backlog'da.

## 2. Hibrit skor (RRF)

BM25 kolu talimatsız istek alıyor. RRF sabiti k = 60, her koldan derinlik 100; BM25'in 0 skorlu
dolgusu atılıyor.

Tüm veriyle eğitilen head'lerle; LiveMCPBench ve MCP-Zero'da tool metninde sunucu adı var:

| Koşu | ToolRet NDCG@10 | ToolRet cat-macro | LiveMCPBench R@5 | LiveMCPBench NDCG@10 | MCP-Zero P@1 | MCP-Zero R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head'ler, hibrit yok | **54.03** | 47.14 | **53.03** | **54.14** | 79.87 | 94.20 |
| hibrit, BM25 ağırlığı 0.1 | 53.38 | **47.41** | 50.36 | 53.92 | 82.02 | 95.06 |
| hibrit, 0.25 | 51.95 | 46.44 | 51.93 | 51.79 | 83.85 | 95.53 |
| hibrit, 0.5 | 49.59 | 44.80 | 45.49 | 47.70 | 84.20 | **96.25** |
| hibrit, 1 (düz RRF) | 46.17 | 41.52 | 40.80 | 43.81 | **84.31** | 96.15 |

Ham Qwen3-Embedding'de de tablo aynı:

| Set | Hibritsiz | Düz RRF |
| --- | ---: | ---: |
| ToolRet NDCG@10 | 51.11 | 45.67 |
| LiveMCPBench R@5 | 50.82 | 39.65 |
| MCP-Zero P@1 | 78.19 | 83.78 |

BM25 yalnız MCP-Zero'da yardım ediyor, çünkü oradaki istekler hedef tool'un açıklamasından
yazılıyor. Gerçek kullanıcı görevlerinde (ToolRet, LiveMCPBench) BM25 dense'in çok gerisinde ve
füzyon listeyi aşağı çekiyor. Hiçbir ağırlık üç sette birden kazandırmıyor. **Varsayılan kapalı**;
`--hybrid`, agent'ın tool açıklamasından istek yazdığı kurulumlar için bir seçenek.

## 3. Uyarlanabilir K

Kural: en iyi skorun `margin` kadar altına kadarki tool'lar alınıyor, en az 1, en çok `max_k`.
Hibrit listede kaç tool alınacağını kosinüsler, hangi tool'ların alınacağını füzyon sırası
belirliyor. Kural bir kez sıralayıp birçok kesimi deneyen betikle (`scripts/adaptive_k_sweep.py`)
ToolRet'te seçildi, MCP setlerinde doğrulandı. Seçim ölçütü önceden tanımlıydı: aynı ortalama K'da
sabit top-k'ya göre en büyük tamlık kazancı.

ToolRet (seçim seti; sabit top-k'nın aynı K'daki değeri arada doğrusal):

| Kural | K | Recall | Precision | Tamlık | Aynı K'da sabit top-k tamlığı |
| --- | ---: | ---: | ---: | ---: | ---: |
| top-5 | 5.00 | 57.39 | 17.59 | 46.69 | — |
| top-8 | 8.00 | 63.02 | 12.34 | 52.41 | — |
| top-10 | 10.00 | 65.59 | 10.38 | 55.06 | — |
| margin 0.10, max 10 | 5.88 | 60.58 | 30.90 | 48.83 | 48.68 |
| margin 0.15, max 10 | 7.35 | 63.62 | 23.49 | 52.57 | 51.36 |
| **margin 0.20, max 10** | 8.29 | 64.95 | 18.35 | 54.34 | 52.85 (**+1.49**) |
| margin 0.25, max 10 | 8.97 | 65.29 | 14.85 | 54.78 | 53.94 |

Held-out MCP setlerinde varsayılan (margin 0.20, max 10):

| Set | K | Recall | Precision | Tamlık | Aynı K'da sabit top-k (Recall / Precision / Tamlık) |
| --- | ---: | ---: | ---: | ---: | --- |
| LiveMCPBench (525 tool, 94 görev) | 7.45 | 59.80 | 28.74 | 35.11 | 57.13 / ~19 / 34.52 |
| MCP-Zero (2.792 tool) | 5.88 | 95.82 | 29.63 | 95.70 | 94.74 / ~16 / 94.63 |

Kazanç recall ve tamlıkta küçük (+0.6 ile +2.7), hassasiyette büyük: tek tool'luk isteklere az,
çok tool'luk isteklere çok tool gidiyor. LiveMCPBench 94 görevlik küçük bir set; bir görev ~1 puan.

## 4. Paketlenmiş head (torch'suz)

- `NumpyHeads`, CLM biçimindeki head'leri numpy ile çalıştırıyor. `.npz` dosyası
  `allow_pickle=False` ile okunuyor ve anahtar/şekil kontrolü var; `make_head`'in bütün ileri geçişi
  destekleniyor.
- CI'da torch yok. Onun yerine commit edilmiş küçük bir golden fixture numpy çıktısını torch'unkiyle
  karşılaştırıyor.
- `toolrank heads export`, `.pt`'den `.npz` üretiyor ve servis ayarlarını (backbone, formatlar,
  kırpma, talimat) dosyanın cfg'sine yazıyor.
- `--clm-ckpt` `.pt`, `.npz` ya da `default` kabul ediyor. `default` için indirme sha256 ile
  doğrulanıyor; adres, barındırma yeri seçilince doldurulacak.

Artifact: `toolrank-heads-qwen3-emb-8b-v0.1.npz`. Tüm veriyle eğitilen head'lerden, fp16, 59.8 MB,
sha256 `f3c10125…72f0`. Model kartı: `docs/heads/MODEL_CARD.md`; eğitim verisinin lisanssız olduğunu
yazıyor.

| | ToolRet (proj. kosinüs min / top-10 örtüşme / top-1 aynı) | LiveMCPBench | MCP-Zero |
| --- | --- | --- | --- |
| fp32 `.npz` ↔ `.pt` | 1.000000 / %100.00 / %99.97 | 1.000000 / %100 / %100 | 1.000000 / %100 / %100 |
| fp16 `.npz` ↔ `.pt` | 1.000000 / %99.96 / %99.66 | 1.000000 / %99.89 / %100 | 1.000000 / %99.97 / %99.93 |

Metrikler:

| Set | `.pt` | fp16 `.npz` |
| --- | ---: | ---: |
| ToolRet NDCG@10 | 54.03 | 54.03 |
| ToolRet cat-macro | 47.14 | 47.13 |
| LiveMCPBench R@5 | 53.03 | 53.03 |
| MCP-Zero P@1 | 79.87 | 79.87 |

Varsayılan talimat üç aday arasından seçildi; iki MCP setinde fp16 head'lerle ölçüldü:

| Talimat | LiveMCPBench R@5 | MCP-Zero P@1 | Ortalama |
| --- | ---: | ---: | ---: |
| "Given a user request, retrieve the tools needed to fulfill it." | 52.46 | 54.91 | 53.68 |
| "Given an agent task, retrieve the MCP tools needed to complete it." | **53.03** | 70.45 | 61.74 |
| **"Given an agent's request for a tool, retrieve the MCP tool that fulfills it."** | 51.59 | **79.87** | **65.73** |

MCP-Zero talimata çok duyarlı: nötr talimatla top-1 25 puan düşüyor. LiveMCPBench'teki fark ise
1–2 görev.

## 5. `toolrank search`

GB10'da, ingest dizininde 1.875 tool var: üç MCP sunucusu, GitHub REST ve Stripe. Arama fp16
head'lerle yapıldı (`TOOLRANK_HEADS`).

- İlk arama (soğuk): indeks 0.90 sn'de kuruldu (1.875 tool projekte edildi), arama 85 ms sürdü.
- Sonraki arama (sıcak): indeks diskten 0.00 sn'de yüklendi, arama 79 ms. Aramanın neredeyse tamamı
  sorgunun gömülmesi.

İlk sonuçlar:

- "open an issue in my repository about the login bug" → `github/issues/create`.
- "refund the last payment of this customer" → Stripe'ın dört iade işlemi ilk beşte.
- "what time is it in Tokyo right now" → 2 tool (`time/get_current_time`, `time/convert_time`).
- "echo back this message" → iki sunucunun `echo`'su.
- "cancel the subscription at the end of the billing period" → Stripe'ın iki "Cancel a subscription"
  işlemi.

Uyarlanabilir K net isteklerde 2 tool, geniş isteklerde 10 tool döndürdü.

## Komutlar

```bash
# GB10, ~/toolrank; ortak bayraklar
E="$HOME/.local/bin/uv run toolrank eval"
Q="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst"
H="--scorer clm $Q --clm-ckpt data/heads/qwen_full_skip_neg0_e5.pt"

# 1. eşlik ve indeksler
$E --data data/toolret $H --out results/parity_toolret_headsfull.json
$E --data data/toolret $H --index faiss --index-dir /tmp/w2_faiss_toolret --out results/w2_toolret_headsfull_faiss.json
$E --data data/toolret $H --batch 1 --out results/w2_toolret_headsfull_numpy_b1.json
$E --data data/toolret $H --batch 1 --index faiss --index-dir /tmp/w2_faiss_toolret --out results/w2_toolret_headsfull_faiss_b1.json
docker compose -f deploy/spark/compose.yaml --profile pg up -d toolrank-pg
export TOOLRANK_PG_DSN=postgresql://toolrank:toolrank@127.0.0.1:55440/toolrank
$E --data data/livemcpbench_server $H --index pgvector --pg-table w2_livemcp --out results/w2_livemcp_srv_headsfull_pg.json
$E --data data/mcp_zero_server --ks 1,5,10,20 $H --index pgvector --pg-table w2_mcpzero --out results/w2_mcpzero_srv_headsfull_pg.json

# 2. hibrit (+ aynıları --scorer dense $Q ile; ağırlıklar 0.1, 0.25, 0.5 için --rrf-weight $w)
$E --data data/toolret $H --hybrid --out results/w2_toolret_headsfull_hybrid.json
$E --data data/livemcpbench_server $H --hybrid --out results/w2_livemcp_srv_headsfull_hybrid.json
$E --data data/mcp_zero_server --ks 1,5,10,20 $H --hybrid --out results/w2_mcpzero_srv_headsfull_hybrid.json

# 3. uyarlanabilir K
~/.local/bin/uv run python scripts/adaptive_k_sweep.py --fixed 4,5,6,7,8,9,10 --margins 0.1,0.15,0.2,0.25,0.3 --max-k 10 -- --data data/toolret $H
for d in livemcpbench_server mcp_zero_server; do
  ~/.local/bin/uv run python scripts/adaptive_k_sweep.py --fixed 3,4,5,6,7,8,9,10 --margins 0.2 --max-k 10 -- --data data/$d $H
done

# 4. paketlenmiş head
~/.local/bin/uv run toolrank heads export data/heads/qwen_full_skip_neg0_e5.pt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
  --dtype float16 --backbone Qwen/Qwen3-Embedding-8B --tool-format documentation --query-format instruct_query --truncate 8192 \
  --instruction "Given an agent's request for a tool, retrieve the MCP tool that fulfills it."
~/.local/bin/uv run python scripts/heads_parity.py --a data/heads/qwen_full_skip_neg0_e5.pt --b <fp16|fp32 .npz> -- --data data/<set> --scorer clm $Q
N="--scorer clm --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --clm-ckpt dist/heads/toolrank-heads-fp16.npz"
$E --data data/livemcpbench_server $N --instruction "<aday>" --out results/w2_livemcp_srv_npz16_i<n>.json   # ve mcp_zero_server

# 5. search (data/demo = Mac'teki data/mytools'un kopyası)
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz ~/.local/bin/uv run toolrank search --data data/demo "open an issue in my repository about the login bug"
```

## Ortam

- GB10: vLLM 0.13.0 (NGC 26.01), Qwen3-Embedding-8B bf16 (8091, `--max-model-len 8192`).
- Head'ler: torch `.pt` GPU'da, `.npz` numpy ile CPU'da.
- Kütüphaneler: faiss-cpu 1.15.1, `pgvector/pgvector:pg17`, psycopg 3.3.6, pgvector-python 0.5.0.
- Embedding cache sıcak.
- Kod: `3fd2017` … `27ed4a0` ve bu commit.

## Sapmalar ve açıklamalar

- **Hibrit** planda bir madde olarak vardı; ölçüm onu varsayılan yapmadı. Kod ve bayrak duruyor.
- **Seçim ve test ayrımı.** Uyarlanabilir K ToolRet'te, talimat ise iki MCP setinde seçildi. Bu
  yüzden talimat tablosu bir held-out ölçümü değil, seçim tablosu.
- **Head'ler yayında değil.** Kod ve artifact hazır, `HEADS_URL` boş. Barındırma yeri seçilince tek
  satırla dolacak.
- **pgvector** yalnız exact tarama yapıyor. Büyük kataloglar için numpy ya da FAISS kullanılmalı.

## Sonraki hafta

Faz 1 Hafta 3: `toolrank serve --mcp` (search_tools / call_tool proxy'si), REST `/v1/search`, kullanım
günlüğü şeması. `build.py`, `toolrank search`'ün JSON çıktısı ve kalıcı indeks bunun temeli.
