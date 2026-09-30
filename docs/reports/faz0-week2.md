# Faz 0 — Hafta 2 raporu (29 Eyl 2026)

## Sonuç (tek cümle)

Sıfır-atış matris koşuldu (ToolRet, 4 tool formatı × CLM / clm-raw / Qwen3-Embedding-8B × w/o,
w/ inst): kapının okunduğu ayarda (w/ inst, cat-macro) CLM'in en iyisi `example_call` ile 4.99,
Qwen3-Embedding-8B `documentation` ile 46.54 (StackOne'ın 0.462'sini 0.3 puan içinde yeniden
üretiyor); head'ler olmadan backbone ~0, yani kapı (≥ 50) tamamen Hafta 3'ün head fine-tune'una
kalıyor; gecikme hedefi (p50 ≤ 100 ms) bf16'da 86 ms ile tutuyor, FP8 backbone head'leri bozmadan
49 ms'ye indiriyor.

## Tablo

ToolRet (44.453 tool, 7.961 sorgu), NDCG@10, hücreler "micro / cat-macro". Micro protokolün
sayısı; cat-macro makalenin "Average"ı ve Faz 0 kapısının okunduğu sütun (aşağıda StackOne).
Sorgu formatları: w/o inst `plain`; w/ inst CLM ve clm-raw için `clm`, Qwen3-Embedding için
`instruct_query`.

| Model | inst | `name_desc` | `schema` | `example_call` | `documentation` |
| --- | :-: | ---: | ---: | ---: | ---: |
| CLM | n | 3.17 / 2.47 | 2.97 / 2.34 | 3.76 / 3.08 | 2.23 / 2.00 |
| CLM | y | 5.38 / 4.17 | 4.34 / 2.94 | **6.64 / 4.99** | 5.09 / 3.28 |
| clm-raw | n | 0.50 / 0.42 | 0.62 / 0.68 | 0.55 / 0.56 | 0.36 / 0.28 |
| clm-raw | y | 0.05 / 0.03 | 0.07 / 0.06 | 0.27 / 0.32 | 0.15 / 0.14 |
| Qwen3-Emb-8B | n | 35.70 / 25.29 | 39.27 / 29.51 | 38.23 / 26.80 | 42.73 / 35.28 |
| Qwen3-Emb-8B | y | 42.56 / 34.54 | 46.00 / 38.36 | 44.60 / 36.88 | **51.11 / 46.54** |

Comprehensiveness@10 aynı düzende:

| Model | inst | `name_desc` | `schema` | `example_call` | `documentation` |
| --- | :-: | ---: | ---: | ---: | ---: |
| CLM | n | 3.93 / 2.81 | 3.50 / 2.67 | 4.70 / 3.32 | 2.91 / 2.30 |
| CLM | y | 7.08 / 5.67 | 5.56 / 3.67 | 8.70 / 6.51 | 6.26 / 4.46 |
| clm-raw | n | 0.26 / 0.25 | 0.26 / 0.38 | 0.29 / 0.33 | 0.15 / 0.08 |
| clm-raw | y | 0.08 / 0.03 | 0.10 / 0.07 | 0.18 / 0.08 | 0.11 / 0.09 |
| Qwen3-Emb-8B | n | 37.09 / 26.12 | 40.18 / 30.32 | 39.05 / 27.28 | 43.54 / 36.50 |
| Qwen3-Emb-8B | y | 43.52 / 35.50 | 47.12 / 39.60 | 45.51 / 37.00 | 51.60 / 47.64 |

Her modelin en iyi w/ inst satırı ve referans olarak makale ayarındaki BM25 (`toolrank compare`):

| Run | dataset | inst | n | NDCG@10 | Recall@10 | Comprehensiveness@10 | NDCG@10 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/example_call/clm | toolret | y | 7961 | 6.64 | 10.26 | 8.70 | 4.99 | 0.341 |
| dense/emb/qwen3-8b/example_call/clm | toolret | y | 7961 | 0.27 | 0.45 | 0.18 | 0.32 | 0.777 |
| dense/emb/qwen3-emb/documentation/instruct_query | toolret | y | 7961 | 51.11 | 62.32 | 51.60 | 46.54 | 0.805 |
| bm25/documentation/concat/nostem | toolret | y | 7961 | 39.27 | 49.49 | 40.70 | 36.41 | 0.42 |

(p50 burada sıcak cache ile sıralama adımı; soğuk gecikme Hafta 1 raporunda.)

Görev bazında, en iyi CLM (`example_call`, w/ inst) ile en iyi Qwen3-Embedding (`documentation`,
w/ inst) arasındaki fark (`scripts/per_task_diff.py`); CLM'in en az ve en çok geride kaldığı üçer
görev:

| Task | n | CLM NDCG@10 | Qwen3-Emb NDCG@10 | CLM − Qwen3 |
| --- | ---: | ---: | ---: | ---: |
| gta | 14 | 11.07 | 10.23 | +0.84 |
| rotbench | 550 | 1.25 | 10.47 | −9.23 |
| toollens | 314 | 0.52 | 10.95 | −10.43 |
| taskbench-daily | 40 | 1.08 | 67.98 | −66.90 |
| gpt4tools | 32 | 10.28 | 80.69 | −70.41 |
| craft-math-algebra | 280 | 5.34 | 93.14 | −87.80 |

CLM yalnız `gta`'da (14 sorgu) önde; en az geride kaldığı görevler Qwen3-Embedding'in de zayıf
olduğu görevler (rotbench, toollens). Fark her yerde ve en çok, tool tanımı ile sorgunun anlam
olarak yakın olduğu görevlerde (craft, gpt4tools) açılıyor.

Format kararı: CLM için `example_call` (her iki ayarda en iyisi; beklendiği gibi eğitim
dağılımına, yani ajan yörüngelerindeki somut çağrılara en yakın format, ama formatlar arası fark
yalnız 1–2 puan). Qwen3-Embedding için `documentation` (her iki ayarda açık ara en iyisi, `schema`'ya
göre +5; makalenin dense protokolü de bu). Hafta 3'ün head fine-tune'u CLM tarafında
`example_call` + `clm` state (w/ inst) ile başlamalı.

## Gecikme ve FP8 (44k tool, batch=1)

Kapı ayarındaki sorgulardan (w/ inst) rastgele 100 tanesi, tek tek, tüm indekse (44.453 tool)
karşı. Soğuk: sorgu vLLM'de kodlanıyor (HTTP + backbone + head'ler + tam top-k); sıcak: sorgu
cache'ten. İndeks her iki durumda cache'ten kuruluyor (`scripts/latency.py`).

| Scorer | Backbone | Soğuk p50 ms | Soğuk p95 ms | Sıcak p50 ms |
| --- | --- | ---: | ---: | ---: |
| CLM (`example_call`) | Qwen3-8B bf16 | 85.9 | 104.0 | 1.8 |
| CLM (`example_call`) | Qwen3-8B FP8 | **49.3** | **59.1** | 1.8 |
| Qwen3-Embedding-8B (`documentation`) | bf16 | 93.7 | 115.2 | 7.4 |

- Faz 0'ın ikincil hedefi (GB10'da state encode + sıralama p50 ≤ 100 ms) bf16'da 85.9 ms ile
  tutuyor (p95 104 ms sınırda); FP8'de 49.3 ms, CLM'in RTX 4090 referansının (58.1 ms) da altında.
  Sentetik setteki 66.7 ms'den farkı daha uzun sorgular (talimat ekli). Sıcakta Qwen3 daha yavaş
  çünkü skor 4096 boyutta; CLM head'leri 512'ye indiriyor.
- FP8 head'leri bozmuyor (`scripts/fp8_agreement.py`): aynı metinlerin bf16 ve FP8 backbone
  vektörleri arasında kosinüs ortalaması 0.9994 (tool) / 0.9992 (sorgu), head'lerden sonra 0.9977 /
  0.9971 (p1 ≥ 0.993). ToolRet'te CLM `example_call` w/ inst: FP8 6.83 / 5.08, bf16 6.64 / 4.99.
  Ama top-10 listeleri 1000 sorguda ortalama %80 örtüşüyor ve top-1 %74 aynı: sıfır-atış skorlar
  birbirine çok yakın, küçük sayısal farklar sıralamayı değiştiriyor. Fine-tune sonrası yeniden
  bakılmalı.
- Korpus gömme FP8'de yalnız %7 hızlı (350 sn vs 378 sn): toplu prefill hesap sınırlı, tek sorgu
  ise bellek bant genişliği sınırlı, bu yüzden FP8'in asıl kazancı sorgu gecikmesinde.
- Ölçüm sırasında GB10'da başka projenin `judge` konteyneri de çalışıyordu; o anki yükü
  bilinmiyor.

```bash
# GB10, ~/toolrank
docker compose -f deploy/spark/compose.yaml --profile fp8 up -d qwen3-8b-fp8
~/.local/bin/uv run python scripts/latency.py --scorer clm --emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --tool-format example_call
~/.local/bin/uv run python scripts/latency.py --scorer clm --emb-url http://127.0.0.1:8092/v1 --emb-model qwen3-8b-fp8 --truncate 2048 --tool-format example_call
~/.local/bin/uv run python scripts/latency.py --scorer dense --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation
~/.local/bin/uv run toolrank eval --data data/toolret --scorer clm --emb-url http://127.0.0.1:8092/v1 --emb-model qwen3-8b-fp8 --truncate 2048 --emb-batch 128 --tool-format example_call --with-inst --out results/toolret_clmfp8_example_call_inst.json
~/.local/bin/uv run python scripts/fp8_agreement.py
```

## Komutlar

```bash
# GB10, ~/toolrank; iki pooling servisi ayakta (deploy/spark/compose.yaml)
tmux new -d -s matrix 'bash scripts/run_matrix.sh 2>&1 | tee -a data/logs/matrix.log'
bash scripts/run_matrix.sh   # ikinci geçiş: boş-sorgu düzeltmesinden önce düşen iki koşuyu yeniden dener

# Mac
scp 'gb10:toolrank/results/toolret_*.json' results/
toolrank compare results/toolret_clm_example_call_inst.json results/toolret_clmraw_example_call_inst.json results/toolret_qwen3emb_documentation_inst.json results/toolret_bm25_nostem_inst.json
uv run python scripts/per_task_diff.py results/toolret_clm_example_call_inst.json results/toolret_qwen3emb_documentation_inst.json --top 3
```

## Ortam

- GB10 (ASUS Ascent GX10), NGC `vllm:26.01-py3` (vLLM 0.13.0). İki pooling servisi de
  `--no-enable-chunked-prefill --max-num-batched-tokens 8192` ile (aşağıda neden); 8090'da prefix
  caching açık (CLM referansı gibi).
- Kırpma: CLM backbone (8090) `--truncate 2048` (CLM referansı), Qwen3-Embedding-8B (8091)
  `--truncate 8192` (sunucunun `--max-model-len`'i). İstemci batch'i `--emb-batch 128`.
- Korpus gömme maliyeti (44k tool; sonuç dosyalarındaki `encoder_tokens` ve `index_s`):

  | Format | CLM backbone (8090) | Qwen3-Embedding-8B (8091) |
  | --- | ---: | ---: |
  | `name_desc` | ~4 dk (ilk deneme takılmaya denk geldi, token kaydı yok) | 1.73M token*, 314 sn |
  | `schema` | 2.83M token*, 529 sn | 2.60M token, 559 sn |
  | `example_call` | 1.68M token, 378 sn | 1.74M token, 388 sn |
  | `documentation` | 7.28M token, 1.370 sn | 7.70M token, 1.505 sn |

  \* w/o inst sorguları dahil (o modelde ilk başarılı `plain` koşusu). Verim ~4.5–5.3k token/sn;
  matrisin tamamı ~1 sa 40 dk. Embedding cache matris sonunda 5.9 GB.
- Kod: `390396d` (matris betiği ve token kaydı), `f068557` (vLLM bayrakları), `aa2e6c2` (boş
  sorgu). Sonuç dosyalarının `config` alanı `emb_url`, `truncate`, `batch` ve `encoder_tokens`
  içeriyor.

## Sapmalar ve açıklamalar

- **vLLM takılması (düzeltildi).** Matrisin ilk koşusunda 8090 motoru 143. istekte durdu: istek
  "Running" görünüyor ama ilerlemiyordu, GPU %0 ve yeni isteklere de cevap yok; istemci 600 sn'lik
  zaman aşımına kadar bekledi. Aynı 128 metinlik batch tek başına yeniden üretildi: 2.650 tokenlık
  (2.048'e kırpılan) bir metin tek başına ya da diğer 127 metin kendi başına sorunsuz geçiyor,
  birlikte gönderilince motor kilitleniyor. Sebep chunked prefill: adım bütçesi (2.048 token) kısa
  istemlerle dolunca uzun istem bölünüyor ve vLLM 0.13'te bu pooling isteği tamamlanmıyor. İlk
  şüpheli olan prefix caching'i kapatmak sorunu çözmedi (geri açıldı); chunked prefill kapatılıp
  bütçe 8.192'ye çıkınca batch 1.8 sn'de geçti ve verim ~3.7k'dan ~5k token/sn'ye çıktı. Kayıp
  ~35 dk.
- **Boş sorgu (düzeltildi).** `mnms_query_17`'nin metni boş, yalnız talimatı var. w/o inst ayarında
  8090 "The decoder prompt cannot be empty" (400) döndürüyor ve CLM tarafındaki w/o inst koşuları
  düşüyordu (Qwen3-Embedding sunucusu boş girdiyi kabul ediyor). Encoder artık boş metni tek
  boşluk olarak gönderiyor (bir token, içeriksiz); sorgu değerlendirmede kalıyor, etkisi micro
  ortalamada en fazla 0.01 puan. Resmi kod da BM25'te boş sorguya `'NONE'` koyuyor.
- **Parametre biçimleri (düzeltildi, matristen önce).** `formats._params` ToolRet tool'larının
  22.849'unda parametre buluyordu (`code` kategorisinde 0); ToolBench/T-Eval'in required/optional
  listeleri, UltraTool'un `doc_arguments` şeması, AppBench, Gorilla ve GTA biçimleri eklendi:
  35.934 tool. Kalan tool'ların dokümanında parametre alanı yok. `example_call` Gorilla'nın gerçek
  `api_call`'ını kullanıyor.
- **StackOne'ın ayarı.** StackOne'ın ToolRet-full tablosu (kendi 109M modelleri 0.544,
  Qwen3-Embedding-8B 0.462) aynı veriyi kullanıyor (44.453 / 7.961); diğer satırları makalenin
  w/ inst tablosundan birebir: NV-Embed-v1 0.427 = 42.71, GritLM-7B 0.411 = 41.13. Bizim
  Qwen3-Embedding-8B `documentation` w/ inst koşumuz cat-macro 46.54 veriyor (micro 51.11): 0.462
  ile 0.3 puan içinde. Yani kapının ayarı makalenin dense protokolü: w/ inst, cat-macro; kapı
  "fine-tune'lu CLM, w/ inst cat-macro ≥ 50". Kaynak:
  https://www.stackone.com/blog/autoresearch-charged-action-search/
- **Qwen3-Embedding talimat biçimi.** `instruct_query` "Instruct: …\nQuery: …" üretiyor; modelin
  kartındaki biçimde "Query:" sonrası boşluk yok. Küçük bir fark, ölçülmedi; StackOne ile uyum bunun
  sonucu değiştirmediğini düşündürüyor.
- **Hafta 1 notunun düzeltmesi.** Koşudan koşuya vektör farklarını prefix caching'e bağlamıştım;
  aynı batch prefix caching kapalıyken iki kez gönderildiğinde de bileşen başına ~3e-3'e varan
  fark çıkıyor, yani kaynak GPU belirlenimsizliği. `faz0-week1.md` ve CLAUDE.md düzeltildi.
- **Ham backbone'da sinyal var mı?** clm-raw her formatta ~0. Sebep yalnızca anizotropi değil:
  vektörler arası ortalama kosinüs 0.87, ama korpus ortalaması çıkarılıp yeniden normalize
  edildiğinde de `name_desc` NDCG@10 0.45 (w/o) / 0.06 (w/ inst) kalıyor. Donmuş Qwen3-8B'nin
  son-token vektöründeki bilgi ancak öğrenilmiş bir dönüşümle kullanılabilir; sıfır-atış head'ler
  bunu soru → cevap için yapıyor (CLM README örneği tutuyor), tool retrieval için yapmıyor.
- **Anotasyon uyarısı (Hafta 3 için).** Eylül 2026'da çıkan bir çalışma (arXiv 2609.08327, Tool-DE
  üzerinde) tek-doğru-cevaplı anotasyonların retriever'ları düşük gösterdiğini ve fine-tune
  kazancının %30–47'sinin değerlendirme artefaktı olabileceğini bildiriyor. ToolRet de sorgu başına
  tek gold küme taşıyor; head fine-tune sonuçları bunu göz önüne alarak okunmalı.

## Sonraki hafta

- Hafta 3 (O5), head fine-tune: ToolRet-train üzerinde, cache'lenmiş backbone vektörleriyle;
  CLM tarafında `example_call` + `clm` state (w/ inst). Değerlendirme kapının ayarında (w/ inst,
  cat-macro) ve held-out görevlerle, anotasyon artefaktına karşı.
- Öneri: aynı head eğitimini Qwen3-Embedding-8B vektörleri üzerinde de koşmak. Planın kapı
  seçeneklerinden biri zaten "Qwen3-Embedding + kendi head'lerimiz"; iki backbone'un cache'i hazır,
  ek maliyet yalnız eğitim. Sıfır-atışta 46.54, yani 50 için +3.5 puan gerekiyor; CLM için +45.
- FP8 backbone üretim için aday: aynı kalite, yarı gecikme. Hafta 3'ün head'leri bf16 vektörlerle
  eğitilip FP8 ile de değerlendirilmeli (sıralama örtüşmesi fine-tune sonrası yeniden ölçülmeli).
- Hafta 2'den açık kalan: O4 `example_call_llm` (tool başına LLM ile üretilmiş örnek çağrı; bir
  üretim modeli gerekiyor).
