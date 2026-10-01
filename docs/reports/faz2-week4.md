# Faz 2 — Hafta 4 raporu (1 Ekim 2026): öğrenme döngüsü simüle trafikte ölçüldü

## Sonuç (tek cümle)

`toolrank learn`, ToolRet kataloğunda (44.453 tool) 5.573 simüle isteğin günlüğünden öğrenince hiç
görmediği 2.388 istekte NDCG@10'u 54,20 → 57,23'e (cat-macro 48,59 → 52,55) çıkarıyor; kazancın
tamamı trafiğin daha önce sorduğu tool'lara gelen yeni isteklerde, başka benchmark'larda unutma yok,
ve aynı fark canlı A/B'de günlükten okunuyor (`mrr` 0,572 → 0,593, karar `promote`) — yani Hafta 3'te
yalnız mekanizması doğrulanan döngü kalite de üretiyor.

## Ölçüler, birimler ve yön

- `NDCG@10 ↑ %`: `toolrank eval` protokolü (ilk 100, tüm katalog), w/ inst. ToolRet satırlarında
  `micro / cat-macro`; MCP setlerinde tek sayı (micro).
- **Ayrılmış**: sorguların rastgele %30'u (tohumlu); hiçbir zaman servis edilmedi, günlüğe girmedi.
  **Sorulmuş tool**: ayrılmış sorgulardan, altın tool'larından en az biri trafikteki bir sorgunun da
  altın tool'u olanlar. **Yeni tool**: hiçbir altın tool'u trafikte sorulmamış olanlar
  (`heldout_new/`). "Sorulmuş tool" sütunu iki rapordan türetildi (micro ortalama sorgu sayısıyla
  doğrusal): `(n·tümü − n_yeni·yeni) / (n − n_yeni)`.
- `log.Recall@5 ↑ %`: `learn`'ün kendi seçim ölçüsü — günlüğün en yeni %20 isteğinde çağrılan tool
  tüm katalogda ilk 5'te mi (başlangıç → seçilen epoch).
- `mrr ↑`: `toolrank ab`'nin ölçüsü — çağrılan tool'un 1/sırası, kolun bütün aramaları üzerinden.

## Simülasyon

Bir benchmark, servis edilen bir katalog yerine geçiyor (`scripts/learn_sim.py`). Sorguların %70'i
"trafik": her biri `toolrank serve`'ün yaptığı gibi yanıtlanıyor (aynı `Retriever`, uyarlanabilir K
0,2 / en çok 10, aynı kullanım günlüğü) ve simüle ajan **gösterilen** altın tool'ları çağırıyor;
gösterilmeyen altın tool hiç çağrılmıyor, yani günlük yalnız servisin zaten yüzeye çıkardığını
öğretebiliyor (gerçekte olduğu gibi). ToolRet'te başlangıç head'leriyle aramaların %77,4'ünde bir
altın tool gösterildi (ilk sırada %45,5, ortalama K 8,27). Sonra günlükte `toolrank learn`, ve
başlangıç ile öğrenilmiş head'ler ayrılmış sorgularda `toolrank eval` ile ölçülüyor.

## Tablo 1 — trafik boyutu (ToolRet kataloğu, varsayılan tarif)

Unutma sütunları: LiveMCPBench ve MCP-Zero (`_server`), NDCG@10 micro.

| Trafik (istek) | Çift | Karar (epoch) | log.Recall@5 ↑ % | Ayrılmış ↑ % (2.388) | Sorulmuş tool ↑ % (1.327) | Yeni tool ↑ % (1.061) | LiveMCPBench ↑ % | MCP-Zero ↑ % |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| — (v0.1 head'leri) | — | — | — | 54,20 / 48,59 | 46,62 | 63,67 | 53,95 | 88,53 |
| 100 | 77 | yayın yok | 86,7 → 86,7 | — | — | — | — | — |
| 300 | 246 | yayın (3) | 87,8 → 93,9 | 55,11 / 48,81 | 47,88 | 64,16 | 54,85 | 88,67 |
| 1.000 | 801 | yayın (2) | 91,9 → 92,5 | 55,27 / 49,45 | 48,53 | 63,70 | 53,89 | 88,58 |
| 3.000 | 2.309 | yayın (2) | 91,6 → 92,6 | 56,64 / 51,82 | 50,78 | 63,98 | 54,66 | 88,45 |
| **5.573** | **4.313** | **yayın (2)** | 91,3 → 93,0 | **57,23 / 52,55** | **51,95** | 63,83 | 56,05 | 88,96 |

Başka bir bölme (seed 1, 5.573 istek → 4.314 çift): ayrılmış 53,97 / 48,02 → 57,49 / 53,02;
sorulmuş tool (1.292) 44,38 → 50,61; yeni tool (1.096) 65,28 → 65,61; LiveMCPBench 55,37, MCP-Zero 88,72.

Görev kırılımı (5.573 istek, seed 0): 35 görevin 32'si yükseldi; en çok az tool'lu, tekrar eden
görevler (autotools-music +25,6, ultratool +18,8, autotools-weather +15,6, autotools-food +13,8,
gta +13,3); üçü düştü (gpt4tools −6,6, apibank −2,7, taskbench-huggingface −1,0).

## Tablo 2 — tarif ve gürültü (ToolRet, 5.573 istek)

| Tarif | Çift | Karar (epoch) | log.Recall@5 ↑ % | Ayrılmış ↑ % | Yeni tool ↑ % | LiveMCPBench ↑ % | MCP-Zero ↑ % |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| varsayılan (3 epoch, lr 1e-5, `--neg 5`) | 4.313 | yayın (2) | 91,3 → 93,0 | 57,23 / 52,55 | 63,83 | 56,05 | 88,96 |
| `--epochs 10` | 4.313 | yayın (2) | 91,3 → 93,0 | 57,33 / 52,80 | 63,90 | 55,83 | 88,84 |
| `--lr 3e-5` | 4.313 | yayın (3) | 91,3 → 91,5 | 57,29 / 53,35 | 63,01 | 55,58 | 88,58 |
| `--neg 0` (yalnız batch içi negatif) | 4.313 | yayın (2) | 91,3 → 93,9 | 55,71 / 50,92 | 64,15 | 56,47 | 88,54 |
| gürültülü ajan (`--noise 0.2`), varsayılan | 4.549 | **yayın yok** | 93,1 → 93,1 | — | — | — | — |
| gürültülü ajan, `--strict` | 3.498 | yayın (3) | 91,0 → 92,6 | 57,22 / 52,45 | 64,38 | 56,13 | 88,67 |

Gürültülü ajan: isteklerin %20'sinde altın tool yerine gösterilen ilk yanlış tool'u çağırıyor ve
çağrı `tool_error` bitiyor (1.051 zayıf pozitif; o isteklerde altın tool "gösterildi, çağrılmadı"
negatifi oluyor).

## Tablo 3 — MCP-Zero kataloğu (2.792 tool, tool başına bir istek; koruma seti LiveMCPBench)

Burada ayrılmış her istek trafikte hiç sorulmamış bir tool'a (838'in 829'u).

| Trafik (istek) | Çift | Karar (epoch) | log.Recall@5 ↑ % | Ayrılmış ↑ % (838) | ToolRet ↑ % (unutma) | LiveMCPBench ↑ % |
| ---: | ---: | --- | --- | ---: | --- | ---: |
| — (v0.1 head'leri) | — | — | — | 89,24 | 54,03 / 47,13 | 53,95 |
| 300 | 288 | yayın yok | 98,3 → 98,3 | — | — | — |
| 1.000 | 952 | yayın (1) | 96,8 → 98,9 | 91,28 | 53,53 / 46,76 | 54,98 |
| 1.954 | 1.857 | yayın (2) | 98,7 → 99,2 | 91,96 | 53,54 / 46,68 | 55,02 |

## Tablo 4 — A/B: ayrılmış sorgular "sonraki dönemin trafiği" olarak

5.573 istekten öğrenilen aday `DATA/heads/candidate.npz`; ayrılmış 2.388 sorgu `--candidate-share 0.5`
ile oynatıldı, `toolrank ab --dry-run` günlüğü okudu.

| Kol | Arama | Çağrıya giden | İlk sırada ↑ | `mrr` ↑ |
| --- | ---: | ---: | ---: | ---: |
| control (v0.1) | 1.220 | 951 | 0,461 | 0,572 |
| candidate | 1.168 | 917 | 0,498 | 0,593 |

Karar: `promote` (fark +0,021, eşik 0,01; kol başına ≥ 100 arama). Aday dosya göründükten 6,5 sn
sonra devredeydi (44.453 tool'un yeniden izdüşümü).

## Komutlar

```bash
# GB10, ~/toolrank; Qwen3-Embedding-8B 8091'de, bütün vektörler önbellekte (encoder tokens 0)
bash scripts/learn_sim.sh                                          # Tablo 1: 100/300/1000/3000/tümü
SIZES=0 SEED=1 NAME=toolret_s1 LEARN="--device cpu" bash scripts/learn_sim.sh   # ikinci bölme
SIZES=0 TAG=e10 LEARN="--epochs 10" bash scripts/learn_sim.sh      # Tablo 2
SIZES=0 TAG=lr3e5 LEARN="--lr 3e-5" bash scripts/learn_sim.sh
SIZES=0 TAG=neg0 LEARN="--neg 0" bash scripts/learn_sim.sh
SIZES=0 NOISE=0.2 bash scripts/learn_sim.sh
SIZES=0 NOISE=0.2 TAG=strict LEARN="--strict" bash scripts/learn_sim.sh
BENCH=data/mcp_zero_server NAME=mcpzero GUARD=data/livemcpbench_server \
  EVALS="data/toolret data/livemcpbench_server" SIZES="300 1000 0" bash scripts/learn_sim.sh   # Tablo 3

# Tablo 4 (elle; A/B günlüğü kirlettiği için ayrı bir dizinde: data/sim/toolret)
export TOOLRANK_HEADS=$PWD/dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz
EMB="--emb-url http://127.0.0.1:8091/v1 --cache-dir .cache/toolrank"
uv run python scripts/learn_sim.py split --bench data/toolret --out data/sim/toolret
uv run python scripts/learn_sim.py traffic --queries data/sim/toolret/traffic.jsonl -- --data data/sim/toolret $EMB
uv run toolrank learn --data data/sim/toolret $EMB --dev data/mcp_zero_server
uv run python scripts/learn_sim.py traffic --queries data/sim/toolret/heldout/queries.jsonl --wait-candidate -- \
  --data data/sim/toolret $EMB --candidate-share 0.5
uv run toolrank ab --data data/sim/toolret --dry-run

# Mac: tablolar results/sim_*.json ve results/learn_sim_*.json raporlarından
uv run toolrank compare results/sim_toolret_base_heldout.json results/sim_toolret_n0_default_heldout.json
```

Sürücünün her adımı gerçek komut: `learn_sim.py traffic` → `toolrank learn --data RUN --out
RUN/learned/<TAG>.npz --dev GUARD` → `toolrank eval --scorer clm --clm-ckpt <head> --tool-format
documentation --query-format instruct_query --with-inst` (ayrılmış, yeni tool ve unutma setleri).

## Ortam

- GB10; Qwen3-Embedding-8B bf16 (NGC vLLM 0.13, `--max-model-len 8192`), önbellek sıcak: hiçbir koşu
  embedding istemedi. Başlangıç head'leri `toolrank-heads-qwen3-emb-8b-v0.1.npz` (fp16).
- Kod: `dc54bce` + bu haftanın değişiklikleri (`learn.state_vectors`, `scripts/learn_sim.*`).
- Süreler: 5.573 arama + günlük 153 sn (44.453 tool, tek iş parçacığı); `learn` 3 epoch × ~13 sn +
  ölçümler ≈ 1 dk (GPU); `learn --dry-run` 2,2 sn (önbellek 872 bin satır, 14,8 GB — anahtarlar önce
  okunuyor, vektörlerin yalnız eşleşenleri).
- Seed 1 eğitimi CPU'da koştu (`--device cpu`): GB10'un belleğini o sırada başka bir koşunun
  reranker'ları doldurmuştu, GPU'da `CUDA out of memory` verdi. Diğer bütün eğitimler GPU'da.

## Sapmalar ve açıklamalar

- **Bu bir simülasyon.** Ajan kusursuz (gösterilen altın tool'u hep çağırıyor, başka hiçbir şeyi
  çağırmıyor) ve istekler benchmark sorguları. ToolRet'te aynı görevin sorguları birbirine benziyor;
  "sorulmuş tool" kazancı (+5,3; seed 1'de +6,2) gerçek trafikte isteklerin ne kadar tekrar ettiğine
  bağlı olacak. Gerçek kullanıcı günlüğüyle ilk tur hâlâ yapılacak iş.
- **Kazanç nerede:** ToolRet'te trafiğin hiç sormadığı tool'lara gelen isteklerde değişiklik yok
  (63,67 → 63,83; seed 1: 65,28 → 65,61) — ne kazanç ne kayıp. MCP-Zero'da ise ayrılmış isteklerin
  hepsi yeni tool'lara ve yine +2,7 var: orada istekler tek kalıptan (Qwen3-8B yazdı), head'ler
  kalıbı öğreniyor. İkisi birlikte: döngü "bu kataloğun bu tür istekleri"ni öğreniyor.
- **Unutma:** replay olmadan en kötü durum MCP-Zero trafiğinden sonra ToolRet'te −0,5 (54,03 → 53,54;
  cat-macro 47,13 → 46,68); ToolRet trafiğinden sonra iki MCP seti de düşmedi. `--replay` bu
  koşularda kullanılmadı.
- **Kaç istek gerekir:** 100 istekte (77 çift, dev 15) ve MCP-Zero'nun 300'ünde (dev 58, başlangıç
  zaten %98,3) hiçbir epoch başlangıcı geçmedi, hiçbir şey yayımlanmadı — doğru davranış. 300 istekte
  (246 çift) küçük (+0,9 micro), binlerde belirgin (+2,4 … +3,0).
- **`log.Recall@5` kazancı küçük gösteriyor:** yalnız altın tool'u zaten gösterilmiş isteklerden
  oluşuyor (başlangıç %91); seçim için yeterli, kazancın büyüklüğü için değil.
- **Varsayılanlar kalıyor:** 10 epoch aynı epoch'u seçti, lr 3e-5 fark yaratmadı. `--neg 0` ise
  ayrılmışta 1,5 puan geride: gösterilip çağrılmayan tool'lar yararlı negatifler (Faz 0'ın
  madenlenmiş negatiflerinin tersine — onlar eşdeğer tool'larla doluydu, bunlar ajanın gözü önünde
  elenmiş olanlar).
- **Yanlış çağrılar:** ajanın yanlış seçimi `tool_error` ile bitiyorsa zayıf pozitifler eğitimi
  bozuyor; varsayılan tarif hiçbir şey yayımlamadı (güvenli taraf), `--strict` ile kazanç tam geri
  geldi (57,22 / 52,45, 3.498 çiftle). Simüle edilmeyen öbür durum — doğru tool, yanlış argüman — için
  zayıf pozitif doğru sinyal; hangisinin baskın olduğunu gerçek günlük gösterecek. Öneri: gerçek
  veride ikisini de koşup `--strict`'i varsayılan yapmayı değerlendirmek.
- A/B'de iki kolun `gold_shown` payı neredeyse aynı (0,780 / 0,785): aday daha çok tool bulmuyor,
  bulduğunu daha yukarı koyuyor (ilk sıra 0,461 → 0,498) ve daha az tool gösteriyor (ortalama K
  8,32 → 7,92).

## Sonraki hafta

- Hafta 5: küme düzeyinde retrieval (birlikte-kullanım istatistiği) ve hiyerarşik routing
  (sunucu → tool, "tool gerekli mi" kapısı).
- Gerçek trafikle ilk `learn` + `ab` turu, veri biriktikçe (kendi Claude Code / Claude Desktop
  kullanımı `data/w3` üzerinden).
