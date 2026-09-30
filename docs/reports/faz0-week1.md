# Faz 0 — Hafta 1 raporu (29 Eyl 2026)

## Sonuç (tek cümle)

ToolRet GB10'a çekildi (44.453 tool, 7.961 sorgu) ve BM25 baseline'ı iki protokolde koşuldu:
protokolümüzün mikro ortalaması (30.29 / 40.74) makaleden 7.97 / 4.28 puan yüksek, ama fark
tamamen raporlama farkı — makalenin BM25s ayarı (stemming yok, `str(doc)`) ve makalenin
"Average" toplamıyla (önce görev, sonra kategori ortalaması) 22.34 / 36.47 çıkıyor (makale
22.32 / 36.46); veri, metrikler ve retrieval doğru, protokol değişmedi.

## Tablo

Protokol (ToolRet'in yayımlanan kodu: tüm korpusta top-100, sorgular üzerinde mikro ortalama) ve
yanında makale usulü ortalama (`cat-macro`), `toolrank compare`:

| Run | dataset | inst | n | NDCG@10 | Recall@10 | Comprehensiveness@10 | NDCG@10 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/documentation/plain | toolret | n | 7961 | 30.29 | 38.06 | 30.02 | 23.94 | 0.332 |
| bm25/documentation/concat | toolret | y | 7961 | 40.74 | 51.12 | 42.21 | 38.40 | 0.475 |
| bm25/documentation/plain/nostem | toolret | n | 7961 | 29.01 | 37.03 | 28.84 | 22.24 | 0.441 |
| bm25/documentation/concat/nostem | toolret | y | 7961 | 39.27 | 49.49 | 40.70 | 36.41 | 0.42 |

Makaleyle karşılaştırma, NDCG@10. Makalenin "Average"ı (Tablo 4 ve 5) üç kategori skorunun düz
ortalaması, her kategori skoru da o kategorideki görevlerin düz ortalaması:
(18.98 + 21.20 + 26.76) / 3 = 22.31 ≈ 22.32. Bu, `cat-macro` sütunu (`EvalReport.category_macro`).
`str(doc)` satırları `scripts/toolret_paper_avg.py` ile; CLI'da bu tool metni yok.

| Ayar | inst | Mikro (protokol) | Cat-macro | Web | Code | Customized |
| --- | :-: | ---: | ---: | ---: | ---: | ---: |
| Makale, BM25s | n | — | 22.32 | 18.98 | 21.20 | 26.76 |
| `str(doc)`, stemming yok (makale ayarı) | n | 29.05 | **22.34** | 19.05 | 21.21 | 26.76 |
| `documentation`, `--no-stem` | n | 29.01 | 22.24 | 18.88 | 21.19 | 26.66 |
| `documentation`, stemming (toolrank varsayılanı) | n | 30.29 | 23.94 | 19.81 | 21.00 | 31.01 |
| Makale, BM25s | y | — | 36.46 | 26.33 | 41.90 | 41.16 |
| `str(doc)`, stemming yok (makale ayarı) | y | 39.30 | **36.47** | 26.33 | 41.91 | 41.16 |
| `documentation`, `--no-stem` | y | 39.27 | 36.41 | 26.17 | 41.90 | 41.16 |
| `documentation`, stemming (toolrank varsayılanı) | y | 40.74 | 38.40 | 27.88 | 41.73 | 45.58 |

Comprehensiveness@10 da tutuyor: `--no-stem` cat-macro 22.05 / 39.03, makale 22.19 / 39.03.

Ara varyantlar, mikro, w/o / w/ inst: `documentation` + stemming yok 29.01 / 39.27; `str(doc)` +
stemming 30.42 / 40.85. Tool metninin kaynağı ~0.1, stemming ~1.3–1.5 puan oynatıyor; gerisi
toplama farkı.

## Komutlar

```bash
# Mac → GB10: izlenen dosyalar (git Mac'te kalır)
git ls-files -z | COPYFILE_DISABLE=1 tar --null -T - --no-xattrs --no-mac-metadata -cf - | ssh gb10 'mkdir -p ~/toolrank && tar -xf - -C ~/toolrank'

# GB10, ~/toolrank
~/.local/bin/uv venv && ~/.local/bin/uv pip install -e ".[dev,data]"
~/.local/bin/uv run toolrank data pull toolret --out data/toolret
~/.local/bin/uv run toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --out results/toolret_bm25.json
~/.local/bin/uv run toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --with-inst --out results/toolret_bm25_inst.json
~/.local/bin/uv run toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --no-stem --out results/toolret_bm25_nostem.json
~/.local/bin/uv run toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --no-stem --with-inst --out results/toolret_bm25_nostem_inst.json
~/.local/bin/uv run python scripts/toolret_paper_avg.py                                   # makale ayarı
~/.local/bin/uv run python scripts/toolret_paper_avg.py --tool-text documentation --stem   # toolrank varsayılanı

# Mac
scp 'gb10:toolrank/results/toolret_bm25*.json' results/
toolrank compare results/toolret_bm25.json results/toolret_bm25_inst.json results/toolret_bm25_nostem.json results/toolret_bm25_nostem_inst.json
```

## Ortam

- Makine: GB10 (ASUS Ascent GX10), Ubuntu 24.04.5 aarch64, 20 çekirdek; BM25 yalnız CPU.
- Python 3.12.13 (uv 0.10.9), bm25s 0.3.11, PyStemmer 3.1.0, numpy 2.5.3, datasets 5.0.1;
  `uv.lock` Mac'tekiyle aynı.
- Veri: `mangopy/ToolRet-Tools` revizyonu `e06c38c7`, `mangopy/ToolRet-Queries` revizyonu
  `b8c76ad3`; çekme 96 sn.
- Kod: ilk koşular `61f7fbf` ile; `cat-macro` sütunlu koşular ve betik bu raporun son hâliyle
  aynı commit'te (ToolRet sorgularına `category` alanı için veri yeniden çekildi, 48 sn).
- Süre: indeks 1.34 sn; sıralama p50 0.33–0.44 ms/sorgu, batch 64. vLLM ve embedding cache bu
  hafta kullanılmadı.

## Sapmalar ve açıklamalar

- **Toplama farkı (asıl neden).** Protokolümüz ToolRet'in yayımlanan kodundaki "Avg" ile aynı:
  sorgu sayısıyla ağırlıklı ortalama. Makalenin tablolarındaki "Average" ise kategori
  ortalamalarının düz ortalaması, her kategori de görevlerin düz ortalaması. BM25'in çok iyi
  olduğu apigen (56.2) ve toolace (57.1) sorguların %25'ini oluşturuyor; 500'den az sorgulu
  görevlerin ortalaması 21.2. Bu yüzden mikro sayı makaledekinden w/o inst 7–8, w/ inst 3–4 puan
  yüksek. Protokol değişmedi. Karar (29 Eyl): raporlar ikisini birlikte taşır; mikro ortalama
  protokolün sayısı, `cat-macro` (makale usulü) yanında (`toolrank compare` sütunu, `toolrank
  eval` tablosunda kategori satırları).
- **BM25 ayarı.** Makalenin BM25s'i stemming kullanmıyor ve `str(doc)` (sözlüğün Python repr'i)
  indeksliyor; bizim varsayılanımız PyStemmer kuruluysa Snowball stemming ve `documentation`
  (JSON). Makale usulü ortalamada bu fark +1.6 / +1.9 puan (çoğu stemming'den; en çok Customized,
  +4.3 / +4.4). `pyproject.toml`'daki "makaledeki gibi stemming" notu yanlıştı, düzeltildi.
- **Sayılar.** Hub'da 44.453 tool / 7.961 sorgu var (makale metni 43k / 7.6k diyor); Hub
  kartlarındaki `num_examples` ile birebir aynı. Kategori bazında makaleyi 0.07 içinde üretmemiz
  makalenin de bu veriyle koştuğunu gösteriyor. Mükerrer id yok, 14.106 gold etiketin hepsi
  korpusta, her sorguda en az bir gold var (ortalama 1.77), tüm relevance değerleri 1.
- **Resmi kod.** `toolret/eval.py`'deki `eval_bm25` bugünkü Hub verisiyle çalışmıyor: tool
  satırlarında `doc` kolonu yok (yalnız `id`, `documentation`) ve `load_tools(task)` görev adını
  kategori doğrulamasına sokuyor. Makale sayıları eski bir şemayla üretilmiş olmalı; `doc`'u
  `documentation`'dan yeniden kurunca (`str(doc)`) aynı sonuç çıkıyor.
- **StackOne referansı.** Faz 0 kapısı (CLM ≥ 0.50; Qwen3-Embedding-8B 0.462) StackOne'ın
  "ToolRet-full" ayarına bağlı ve o sayının hangi toplamayla hesaplandığı belli değil. Kapıdan
  önce netleşmeli: aynı BM25 koşusunda iki toplama arasındaki fark 8 puan.

## Spark (GB10): servisler, CLM smoke testi ve gecikme

İki pooling servisi GB10'da NGC konteynerinde çalışıyor. CLM yolu uçtan uca doğru: CLM
README'sindeki örnek sıralama aynı çıkıyor, olasılık 0.994 (referans 0.997). Batch-1 soğuk
gecikme p50 66.7 ms (Faz 0 hedefi ≤ 100 ms). Ölçüm sırasında embedding cache'inin hiç
yazılmadığı ortaya çıktı; `03047f1` ile düzeltildi.

- Servisler: `deploy/spark/compose.yaml`, ağırlıklar önceden `hf download` ile paralel indirildi
  (Qwen3-8B 2.5 dk, Qwen3-Embedding-8B 2.3 dk), açılış ~90 sn/servis. GPU belleği: qwen3-8b
  20.8 GB, qwen3-emb 21.7 GB; yanında başka projenin `judge` konteyneri 12.8 GB. Toplam
  `--gpu-memory-utilization` 0.52.
- 8090 smoke testi (Mac'ten, Tailscale): 4096 boyut, L2 norm 1.0000, `float` ve `base64` birebir
  aynı, pooling LAST. Ham backbone vektörleri çok anizotropik: bir soru ile alakasız bir tool
  arasındaki kosinüs 0.86.
- CLM head'leri: `CLM_v0.1-8B.pt` (18.9M parametre, width 1536, depth 3, LayerNorm), GB10
  GPU'sunda çalışıyor (torch 2.14.0+cu130; sm_121 sorunsuz). Mimari, embedder ve metin düzeni
  referans repodaki (`Contrastive-LM/CLM`: `heads.py`, `embedder.py`, `schema.py`) ile birebir.
- Smoke testi `clm-serve` yerine `toolrank eval --scorer clm` ile yapıldı (handoff O2'deki gibi):
  10 tool'luk sentetik alt küme ve 300 tool'luk tam sentetik set.

Sentetik set, w/o inst, `name_desc` (`dense/emb/qwen3-8b` satırı clm-raw, yani head'siz CLM):

| Run | dataset | inst | n | NDCG@10 | Recall@10 | Comprehensiveness@10 | NDCG@10 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/name_desc/plain | synthetic | n | 200 | 63.18 | 98.50 | 97.50 | — | 0.083 |
| dense/emb/qwen3-emb/name_desc/plain | synthetic | n | 200 | 59.10 | 91.25 | 88.50 | — | 6.457 |
| dense/emb/qwen3-8b/name_desc/plain | synthetic | n | 200 | 2.82 | 5.75 | 4.00 | — | 0.034 |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/name_desc/plain | synthetic | n | 200 | 16.66 | 31.50 | 27.50 | — | 6.364 |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/name_desc/plain | synthetic10 | n | 20 | 80.48 | 100.00 | 100.00 | — | 9.078 |

Head'ler ham backbone'u 6 kat iyileştiriyor, ama Qwen3-Embedding'in çok gerisinde kalıyor. Bu bir
hata değil (README örneği tutuyor): CLM'in eğitimi soru → cevap çiftleri (60M Nemotron DQA) ve
ajan yörüngelerinden bağlam → verilen karar çiftleri; `name_desc` ise bir cevap değil, bir tanım.
Format sorusu Hafta 2'nin matrisinde (en yakın aday `example_call`). Sentetik set bu karşılaştırma
için zayıf bir vekil; buradaki sayılar yalnızca yolun çalıştığını gösteriyor.

Gecikme, batch=1, ilk 100 sentetik sorgu:

| Scorer | Cache | p50 ms | p95 ms |
| --- | --- | ---: | ---: |
| CLM (8090 + head'ler) | yok (`--cache-dir ''`) | 66.7 | 71.9 |
| CLM | sıcak | 0.25 | 0.26 |
| Qwen3-Embedding-8B (8091) | yok | 67.7 | 73.3 |
| Qwen3-Embedding-8B | sıcak | 0.06 | 0.07 |

Soğuk gecikmenin neredeyse tamamı tek bir 8B ileri geçişi. bf16'da ~16 GB ağırlık ve GB10'un
~273 GB/s bellek bant genişliği ~60 ms'lik bir taban demek; FP8 bunu yarıya indirebilir (Hafta 2
maddesi). Skor adımı 300 tool'da önemsiz, 44k tool'da da birkaç ms mertebesinde kalır.

```bash
# GB10, ~/toolrank
.venv/bin/hf download Qwen/Qwen3-8B & .venv/bin/hf download Qwen/Qwen3-Embedding-8B & wait
.venv/bin/hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --local-dir ~/.cache/clm
~/.local/bin/uv pip install -e ".[dev,data,clm]"
docker compose -f deploy/spark/compose.yaml up -d
~/.local/bin/uv run toolrank data synth --out data/synthetic
~/.local/bin/uv run toolrank data synth --out data/synthetic10 --n-tools 10 --n-queries 20
E8090="--emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048"
E8091="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb"
~/.local/bin/uv run toolrank eval --data data/synthetic10 --scorer clm $E8090 --out results/synthetic10_clm.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer clm $E8090 --out results/synthetic_clm.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer dense $E8090 --out results/synthetic_clmraw.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer dense $E8091 --out results/synthetic_qwen3emb.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer bm25 --tool-format name_desc --out results/synthetic_bm25_name_desc.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer clm $E8090 --batch 1 --limit 100 --cache-dir '' --out results/lat_clm_cold.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer clm $E8090 --batch 1 --limit 100 --out results/lat_clm_warm.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer dense $E8091 --batch 1 --limit 100 --cache-dir '' --out results/lat_qwen3emb_cold.json
~/.local/bin/uv run toolrank eval --data data/synthetic --scorer dense $E8091 --batch 1 --limit 100 --out results/lat_qwen3emb_warm.json
```

Ortam: NGC `nvcr.io/nvidia/vllm:26.01-py3` (vLLM 0.13.0, CUDA 13.1), sürücü 580.178.04. Kod
`71bc749`; sıcak ölçümler cache düzeltmesiyle (`03047f1`).

Notlar:
- Embedding cache hatası: `EmbeddingCache` `__len__` tanımladığı için boş cache `False` sayılıyor
  ve `if self.cache:` kontrolleri onu hiç kullanmıyordu; "bir kez göm" iş akışı fiilen
  çalışmıyordu. Düzeltmeden sonra aynı sıcak koşu 66.7 ms yerine 0.25 ms.
- vLLM vektörleri bit düzeyinde tekrarlanabilir değil: aynı batch iki kez kodlandığında bileşen
  başına ~3e-3'e varan farklar çıkıyor (GPU belirlenimsizliği ve batch bileşimi; prefix
  caching'den bağımsız, Hafta 2'de test edildi). Sentetikteki NDCG@10 16.69 ↔ 16.66 farkı bundan.
  Sayıların tekrarlanabilirliğini embedding cache sağlıyor; silinmemesi için bir sebep daha.

## Sonraki hafta

- `formats._params` ToolRet'in parametre şekillerinin çoğunu tanımıyor: 44.453 tool'un yalnız
  22.849'unda parametre buluyor, `code` kategorisinde hiç (0 / 3.794); `required_parameters` /
  `optional_parameters`, `doc_arguments` gibi şekiller atlanıyor. `schema` ve `example_call` bu
  tool'larda parametresiz kalıyor; Hafta 2'deki format matrisinden önce düzeltilmeli.
- Hafta 2 (O3): ToolRet korpusunu GB10'da iki modelle ve dört formatta gömmek (cache artık
  çalışıyor) ve zero-shot matris; CLM için önce `example_call` formatı ve `clm` state'i (w/ inst).
- Gecikme: FP8 backbone (bf16'daki ~60 ms'lik bant genişliği tabanına karşı) ve 44k tool'da
  sıcak/soğuk ölçüm.
