# Faz 0 — Hafta 3 raporu (29 Eyl 2026)

## Sonuç (tek cümle)

Head fine-tune ToolRet-train'in 60K'lık alt kümesiyle yapıldı: kapı ayarında (w/ inst, cat-macro)
CLM 4.99'dan 17.68'e çıkıyor ama kapıdan (50) çok uzak; Qwen3-Embedding-8B üzerine sıfır-atış
kalitesinden başlayan (skip) head'ler yalnız in-batch negatiflerle 46.55'ten 47.23'e çıkıyor
(kapıya −2.8), veri setinin mined negatifleri ise benchmark'ı 10 puana kadar düşürüyor.

## Tablo

Kapı ayarı: ToolRet, w/ inst, NDCG@10 (resmi `toolrank eval --clm-ckpt` koşuları, `toolrank compare`):

| Run | dataset | inst | n | NDCG@10 | Recall@10 | Comprehensiveness@10 | NDCG@10 cat-macro | Comprehensiveness@10 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/example_call/clm | toolret | y | 7961 | 6.64 | 10.26 | 8.70 | 4.99 | 6.51 | 0.341 |
| clm[clm_60k_lr1e-2_negf95]/emb/qwen3-8b/example_call/clm | toolret | y | 7961 | 28.88 | 38.10 | 30.47 | 17.68 | 19.70 | 0.691 |
| dense/emb/qwen3-emb/documentation/instruct_query | toolret | y | 7961 | 51.11 | 62.32 | 51.60 | 46.54 | 47.64 | 0.805 |
| clm[qwen_60k_skip_neg0]/emb/qwen3-emb/documentation/instruct_query | toolret | y | 7961 | 52.55 | 64.12 | 53.21 | **47.23** | 47.98 | 1.533 |
| bm25/documentation/concat/nostem | toolret | y | 7961 | 39.27 | 49.49 | 40.70 | 36.41 | 39.03 | 0.42 |

Satırlar: sıfır-atış CLM, fine-tune'lu CLM, sıfır-atış Qwen3-Embedding-8B, Qwen3-Embedding-8B +
eğitilmiş skip head'ler, makale ayarında BM25.

Koşular (`results/finetune_*.json`): val = eğitim setinden ayrılan 2.000 çiftte recall@10 (epoch
seçimi buna göre); ToolRet micro / cat-macro seçilen epoch'ta. lr, negatif ve başlangıç dışında
hepsi 5 epoch, batch 512.

CLM backbone (Qwen3-8B, `example_call`, `clm` state):

| Koşu | Çift | Başlangıç | lr | Negatif | Val | ToolRet micro / cat-macro |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| sıfır-atış | — | CLM_v0.1-8B | — | — | 0.193 | 6.64 / 5.00 |
| pilot (3 epoch) | 15K | CLM_v0.1-8B | 1e-4 | 15 mined | 0.380 | 10.83 / 9.37 |
| | 15K | CLM_v0.1-8B | 1e-3 | 15 mined | 0.709 | 19.20 / 14.64 |
| | 15K | CLM_v0.1-8B | 3e-3 | 15 mined | 0.787 | 21.98 / 16.11 |
| | 15K | CLM_v0.1-8B | 1e-2 | 15 mined | 0.836 | 24.83 / 16.65 |
| batch 1024 | 15K | CLM_v0.1-8B | 3e-3 | 15 mined | 0.751 | 20.92 / 15.44 |
| 10 epoch | 15K | CLM_v0.1-8B | 3e-3 | 15 mined | 0.834 | 23.89 / 16.19 |
| in-batch | 15K | CLM_v0.1-8B | 3e-3 | yok | 0.747 | 21.72 / 15.47 |
| sıfırdan head | 15K | rastgele | 1e-3 | 15 mined | 0.034 | 0.14 / 0.02 |
| | 60K | CLM_v0.1-8B | 3e-3 | 15 mined | 0.884 | 26.98 / 16.95 |
| | 60K | CLM_v0.1-8B | 1e-2 | 15 mined | 0.924 | 28.96 / 17.01 |
| **filtre 0.95** | 60K | CLM_v0.1-8B | 1e-2 | 15 mined, filtreli | 0.929 | **28.88 / 17.67** |

Qwen3-Embedding-8B backbone (`documentation`, `instruct_query`), skip head (x + MLP(x), başlangıçta
kimlik):

| Koşu | Çift | lr | Negatif | Val | ToolRet micro / cat-macro |
| --- | ---: | ---: | --- | ---: | ---: |
| sıfır-atış (epoch 0) | — | — | — | 0.894 | 51.11 / 46.55 |
| | 15K | 1e-3 | 15 mined | 0.894 | çöküş, epoch 0 seçildi |
| | 15K | 3e-4 | 15 mined | 0.906 | 44.28 / 37.43 (epoch 1'de çöküp toparlanıyor) |
| | 15K | 3e-5 | 15 mined | 0.959 | 44.92 / 36.08 |
| | 15K | 1e-5 | 15 mined | 0.951 | 48.59 / 41.73 |
| yalnız sorgu tarafı | 15K | 1e-5 | 15 mined | 0.932 | 50.19 / 44.17 |
| sıfırdan CLM head | 15K | 3e-3 | 15 mined | 0.012 | ~0 |
| | 60K | 1e-5 | 15 mined | 0.969 | 45.81 / 36.19 |
| filtre 0.95 (%87 kaldı) | 60K | 1e-5 | filtreli | 0.969 | 46.89 / 37.24 |
| filtre 0.85 (%66 kaldı) | 60K | 1e-5 | filtreli | 0.969 | 47.24 / 37.83 |
| filtre 0.75 (%43 kaldı) | 60K | 1e-5 | filtreli | 0.969 | 47.90 / 39.08 |
| **in-batch** | 60K | 1e-5 | yok | 0.948 | **52.54 / 47.20** |
| in-batch, tüm veri (2 epoch) | 206K | 1e-5 | yok | 0.951 | 52.73 / 46.92 |
| in-batch, tüm veri (5 epoch) | 206K | 1e-5 | yok | 0.967 | 54.03 / 47.13 |

Tüm veri (sonradan eklendi, 21:20): kalan 146K çift de gömüldü (143.275 metin, 17.6M token, 60 dk).
Veri 3.4 kat artınca micro 52.5'ten 54.0'a çıkıyor, cat-macro ~47.2'de kalıyor. Resmi değerlendirme
`results/toolret_qwen3emb_headsfull_documentation_inst.json`: 54.03 / 47.14.

## Komutlar

```bash
# GB10, ~/toolrank
toolrank data pull toolret-train
~/.local/bin/uv run python train/embed_pairs.py --emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --tool-format example_call --query-format clm --n-train 60000
~/.local/bin/uv run python train/embed_pairs.py --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation --query-format instruct_query --n-train 60000

# en iyi CLM
~/.local/bin/uv run python train/finetune_heads.py --emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --tool-format example_call --query-format clm --n-train 60000 --init ~/.cache/clm/CLM_v0.1-8B.pt --epochs 5 --lr 1e-2 --neg-filter 0.95 --name clm_60k_lr1e-2_negf95
# en iyi Qwen3-Embedding + head
~/.local/bin/uv run python train/finetune_heads.py --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation --query-format instruct_query --n-train 60000 --epochs 5 --skip --lr 1e-5 --neg 0 --name qwen_60k_skip_neg0

# resmi değerlendirme
~/.local/bin/uv run toolrank eval --data data/toolret --scorer clm --emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --tool-format example_call --with-inst --clm-ckpt data/heads/clm_60k_lr1e-2_negf95.pt --out results/toolret_clmft60kf_example_call_inst.json
~/.local/bin/uv run toolrank eval --data data/toolret --scorer clm --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst --clm-ckpt data/heads/qwen_60k_skip_neg0.pt --out results/toolret_qwen3emb_heads60k_documentation_inst.json
```

Diğer koşuların bayrakları `results/finetune_<ad>.json` içindeki `args` alanında.

## Ortam

- GB10, iki pooling servisi (Hafta 2'deki bayraklarla). Head eğitimi GB10 GPU'sunda: 60K çift ×
  5 epoch ~1 dk, vektörler cache'ten.
- Gömme (60K + 2K çift, iki backbone aynı anda): CLM 66.030 metin / 7.0M token / 56 dk, Qwen3
  73.070 metin / 7.5M token / 67 dk. Maliyetin neredeyse tamamı istekler (talimatla birlikte);
  tool metinlerinin ~%72'si korpus cache'inden geldi.
- Kod: `8eb8e3f` (yükleyici), `fe6e7a6` (eğitici).

## Sapmalar ve açıklamalar

- **Veri ve sızıntı.** ToolRet-train 208.826 çift; her birinde 1–8 pozitif ve 15 mined negatif, tool
  metinleri korpusla aynı JSON biçiminde, `prompt` ToolRet tarzı bir talimat. 776 isteğin metni bir
  test isteğiyle birebir aynı (yalnız talimat farklı); bunlar eğitimden çıkarıldı. Görev etiketi
  yok, bu yüzden plandaki "held-out task" yapılamadı. Pozitiflerin tamamı korpus tool'u ve testin
  7.626 gold tool'unun 2.717'si (%36) eğitimde pozitif: eğitim korpusla aynı alanda, bu da
  fine-tune sayılarını lehte etkiliyor.
- **CLM.** Referans head'lerden başlamak şart: sıfırdan head'ler öğrenmiyor (Hafta 2'deki bulguyla
  uyumlu: ham backbone geometrisi retrieval'a uygun değil). Yüksek lr (1e-2) en iyisi; 15K'dan
  60K'ya micro 24.8 → 29.0, ama cat-macro ~17'de kalıyor: kazanç eğitimde sık görülen büyük
  görevlerde. Mined negatifler az da olsa yardımcı, 0.95 filtresi +0.7.
- **Qwen3-Embedding, skip head.** Başlangıçta sıfır-atışı birebir veriyor (51.11 / 46.55). Yüksek
  lr'de çöküyor: birim normlu girdinin yanında, sıfırla başlatılan 4096 genişlikli çıkış katmanı
  Adam'ın parametre başına ~lr'lik adımlarıyla birkaç adımda kimliği bastırıyor; lr 1e-5 kararlı.
  Sıfırdan CLM mimarisi (512 boyutlu) öğrenmiyor.
- **Mined negatifler Qwen3'e zarar veriyor.** Veri setinin 15 negatifiyle cat-macro 46.55'ten
  36.19'a düşüyor; NV-Retriever tarzı filtre (başlangıç modelinin pozitife yakın bulduğu negatifleri
  atmak) yalnız biraz yardımcı ve negatif azaldıkça sonuç düzenli iyileşiyor (0.95 → 0.75), hiç mined
  negatif olmadan +0.65. Arasında işlevsel olarak eşdeğer tool'lar olduğu kesin (Tool-DE çalışmasının
  "tek doğru cevap" uyarısı), ama sıkı filtrelerin de zarar vermesi ayrıca bir dağılım kaymasına
  işaret ediyor.
- **Val ile benchmark ayrışıyor.** Eğitim setinden ayrılan çiftlerde recall@10 her koşuda artıyor
  (Qwen3'te 0.97'ye), benchmark ise mined negatiflerle düşüyor; mined negatifli her Qwen3 koşusunda
  benchmark'ın en iyi noktası epoch 0. Yani eğitim dağılımından alınan doğrulama, kapı metriği için
  vekil değil. Benchmark eğrisi seçimde kullanılmadı; bu yüzden raporlanan sayılar val seçimine göre.
- **Plandan farklar.** Negatifler BM25 ile madenlenmedi: veri setinin kendi negatifleri zaten
  zarar verdiği için ek madenlemeye geçilmedi. Replay (Nemotron) yapılmadı: CLM kapıdan bu kadar
  uzakken önceliği düşük. Tüm ToolRet-train yerine 60K alt küme kullanıldı; Qwen3 için tüm verinin
  gömülmesi sürüyor.

## Sonraki hafta

- Tüm ToolRet-train ile eğitim yapıldı (yukarıda): cat-macro değişmedi, daha fazla veri kapıyı
  kapatmıyor.
- Hafta 4 (O6): MCP-Zero ve LiveMCPBench. LiveMCPBench koşuldu (`faz0-gate.md`): ToolRet-train'le
  eğitilen head'ler alan dışında da kazandırıyor (Recall@5 49.0 → 52.3).
- Kapı raporu (`faz0-gate.md`): eldeki kanıt CLM yerine "Qwen3-Embedding + kendi head'lerimiz"
  yönünü gösteriyor; kapının kendisi (≥ 50) henüz tutmuyor. Zaman kalırsa Qwen3-Embedding LoRA
  fine-tune'u (planda Hafta 4'ün opsiyonel maddesi) aynı veriyle.
