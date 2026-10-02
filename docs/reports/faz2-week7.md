# Faz 2 — Hafta 7 raporu (2 Ekim 2026): LoRA'lı backbone varsayılan oluyor, ikinci aşama ürüne giriyor

## Sonuç (tek cümle)

Diğer oturumun LoRA ile eğittiği Qwen3-Embedding-8B, FP8'de de bf16 kadar iyi (ToolRet 59,02 / 54,53,
LiveMCPBench NDCG@10 55,34, MCP-Zero top-1 87,71) ve v0.1 head'leri onun üstünde 1–2 puan kaybettiriyor;
bu yüzden paketin varsayılanı head'siz LoRA'lı backbone oldu (`toolrank-emb-v0.2`, Docker / compose /
Helm'de FP8), head'ler yalnız eğitildikleri temel modelde uygulanıyor, ve `search` / `serve` isteğe
bağlı ikinci aşamayı (`--rerank cross | jev`) aldı; ağırlıklar henüz Hub'da değil, yükleme bekliyor.

## Ölçüler, birimler ve yön

`toolrank eval` protokolü, w/ inst; `NDCG@10 ↑ %` (ToolRet'te micro / cat-macro), `Recall@5 ↑ %`,
`P@1 ↑ %` (MCP-Zero'nun top-1'i). Kosinüs ve örtüşme `scripts/fp8_agreement.py`'den.

## Tablo 1 — LoRA'lı backbone: bf16, FP8, ve v0.1 head'leriyle

| Düzen | ToolRet NDCG@10 ↑ % | LiveMCPBench NDCG@10 ↑ % | LiveMCPBench Recall@5 ↑ % | MCP-Zero P@1 ↑ % (seçim seti) |
| --- | --- | ---: | ---: | ---: |
| Qwen3-Embedding-8B + v0.1 head'leri (bugünkü ürün) | 54,03 / 47,13 | 53,95 | 53,03 | 79,87 |
| LoRA, bf16 | 58,90 / 54,36 | 55,74 | 52,06 | 88,57 |
| **LoRA, FP8** | **59,02 / 54,53** | 55,34 | 52,06 | 87,71 |
| LoRA + v0.1 head'leri | 58,71 / 52,94 | 53,52 | 48,18 | 87,46 |

FP8 ile bf16 arasında (LoRA, ToolRet'in 1.000 sorgusu): omurga kosinüsü ortalama 0,9974 (tool) /
0,9971 (sorgu), ilk 10 örtüşmesi %95,4, aynı ilk sıra %94,0.

## Tablo 2 — README tablosu (yeni iki satır)

| Retriever | ToolRet NDCG@10 ↑ % | cat-macro ↑ % | LiveMCPBench Recall@5 ↑ % | MCP-Zero top-1 ↑ % |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-8B + toolrank heads v0.1 | 54,03 | 47,13 | 53,03 | 79,87 |
| Qwen3-Embedding-8B in FP8 + toolrank heads v0.1 | 53,94 | 47,27 | 53,48 | 79,51 |
| toolrank backbone v0.2 (Qwen3-Embedding-8B + LoRA) | 58,90 | 54,36 | 52,06 | 88,57 |
| toolrank backbone v0.2 in FP8 (the default) | 59,02 | 54,53 | 52,06 | 87,71 |

## Ne değişti

- **Backbone kaydı** (`build.BACKBONES`): servis adı → ağırlıklar ve paketli head'lerin ona ait olup
  olmadığı. Varsayılan servis adı `toolrank-emb-v0.2` (adda sürüm var, çünkü embedding önbelleği adla
  anahtarlanıyor). Önbellekteki head'ler yalnız `qwen3-emb` / `qwen3-emb-fp8`'de (ve bilinmeyen adlarda,
  eskisi gibi) yükleniyor; `TOOLRANK_HEADS` her zaman geçerli. `toolrank learn` başlangıcı da aynı
  kuralla seçiyor (LoRA'da sıfırdan skip head, yani epoch 0 = backbone).
- **Dağıtım:** birleşik imajın giriş betiği (`TOOLRANK_BACKBONE` / `_REVISION`, temel model ayrı adla),
  `deploy/docker/compose.yaml`, Helm chart'ının `embedding.backbone` değeri; hepsi FP8 ile
  `yasinyaman/toolrank-emb-8b@v0.2`'yi servis ediyor. Temel modele dönüş tek ayar.
- **Yayın:** `docs/backbone/MODEL_CARD.md` (tarif, seçim seti uyarısı, lisans, sayılar) ve
  `scripts/publish_backbone.py` (GB10'da kuru çalıştırma geçti: 10 dosya, `model.safetensors` 16,4 GB,
  sha256 `53789cff…`). `BACKBONE_PUBLISHED = False` iken `release_check` sürümü reddediyor.
- **İkinci aşama:** `search` / `serve --rerank cross --rerank-emb-url …` ya da `--rerank jev`; ilk 20
  tool, dokümantasyon 3.000 karakter. Kaç tool döneceği ilk aşamanın kosinüsünden (uyarlanabilir K),
  sıra ikinci aşamadan. İki reranker'a paylaşılan duruma yazmayan `rank_pairs` eklendi (sunucuda
  eşzamanlı aramalar `last_base` üzerinde yarışıyordu).

## Ek (2 Ekim, öğleden sonra) — kendi dev setimiz ve LoRA için ne söylediği

**Sonuç:** `toolrank data gen-queries` ile hiçbir benchmark'la sorgu paylaşmayan seçim setleri
üretildi (GitHub + Stripe + üç küçük kaynak, 1.862 tool; Qwen3-8B'nin yazdığı istekler) ve
gösterdikleri şu: LoRA'lı backbone'un kazancı **birden çok tool gerektiren görevlerde** — iki
tool'luk 800 görevde NDCG@10 71,04 → 82,95, üç tool'luk 498 görevde 55,21 → 75,40 (ikisi de
p < 0,0001); **tek tool'luk isteklerde** temel modelle başa baş, işlemi söylemeyen "durum"
isteklerinde ilk 5'te 2,5 puan geride (p = 0,003). İlk iki set (tek tool) yalnız ikinci yarıyı
gösterdiği için öğleden sonra "LoRA bağımsız setlerde kazanmıyor" diye yazılmıştı; çok tool'luk
setler eklenince resim tamamlandı. Kullanıcı LoRA'yı varsayılan bıraktı.

Tablo 3 — dev setleri (w/ inst; `P@1 ↑ %`, `Recall@5 ↑ %`, `NDCG@10 ↑ %`):

| Model | `dev_w3` (1.000 istek: görev / adım / hedef) | `dev_w3_sit` (598 istek: "durum") |
| --- | --- | --- |
| BM25 (talimatsız) | 54,90 / 81,00 / 71,22 | 40,13 / 67,89 / 58,20 |
| Qwen3-Embedding-8B | 84,20 / 99,00 / 92,99 | 76,92 / 97,99 / 89,67 |
| + v0.1 head'leri | 84,20 / 98,80 / 92,98 | 77,26 / 98,16 / 89,86 |
| LoRA'lı backbone | 83,50 / 98,80 / 92,49 | 76,25 / 95,48 / 87,71 |
| LoRA + v0.1 head'leri | 81,30 / 98,10 / 91,18 | 69,90 / 93,65 / 84,37 |

Tablo 4 — temel model ile LoRA, eşli karşılaştırma (yalnız birinin bulduğu istek sayısı; iki
yönlü işaret testi):

| Set | İlk 1: yalnız temel / yalnız LoRA | p | İlk 5: yalnız temel / yalnız LoRA | p |
| --- | ---: | ---: | ---: | ---: |
| `dev_w3` (1.000) | 76 / 69 | 0,62 | 6 / 4 | 0,75 |
| `dev_w3_sit` (598) | 56 / 52 | 0,77 | 19 / 4 | 0,003 |
| LiveMCPBench (94) | 10 / 14 | 0,54 | 5 / 5 | 1,00 |

Tablo 5 — LoRA'nın temel modele göre kazancı, sete göre (NDCG@10 puanı; MCP-Zero'da top-1):

| Set | Niteliği | Temel → LoRA |
| --- | --- | --- |
| ToolRet | eğitim dağılımı (aynı görevlerin eğitim çiftleri) | 51,11 → 58,90 (+7,8) |
| MCP-Zero top-1 | seçim seti; istekler iki satırlık `server: … tool: …` kalıbında | 78,19 → 88,57 (+10,4) |
| LiveMCPBench | bağımsız; doğal dilde çok adımlı görevler | 53,74 → 55,74 (+2,0; anlamlı değil) |
| `dev_w3_multi2` | bağımsız; OpenAPI kataloğu, iki tool'luk görevler | 71,04 → 82,95 (+11,9) |
| `dev_w3_multi3` | bağımsız; aynı katalog, üç tool'luk görevler | 55,21 → 75,40 (+20,2) |
| `dev_w3` | bağımsız; aynı katalog, doğal dilde tek tool'luk istekler | 92,99 → 92,49 (−0,5) |
| `dev_w3_sit` | bağımsız; aynı katalog, işlemi söylemeyen istekler | 89,67 → 87,71 (−2,0) |

Tablo 6 — çok tool'luk görevler (`--tools-per-request`; bütün tool'lar altın; `NDCG@10 ↑ %`,
`Comprehensiveness@10 ↑ %` = bütün tool'lar ilk 10'da, uyarlanabilir K ile `K@cut ↓` / `C@cut ↑ %`):

| Model | `dev_w3_multi2` (800 görev, 2 tool) | `dev_w3_multi3` (498 görev, 3 tool) |
| --- | --- | --- |
| BM25 (talimatsız) | 54,20 / 50,25 | 47,76 / 25,50 |
| Qwen3-Embedding-8B | 71,04 / 72,12 (K 9,86 / 71,88) | 55,21 / 33,94 (K 9,88 / 33,94) |
| + v0.1 head'leri | 71,69 / 74,38 (K 9,50 / 72,50) | 57,28 / 35,74 (K 9,64 / 34,54) |
| **LoRA'lı backbone** | **82,95 / 89,50** (K 9,12 / 87,88) | **75,40 / 64,46** (K 9,47 / 63,05) |
| LoRA + v0.1 head'leri | 80,82 / 86,88 (K 9,82 / 86,75) | 72,92 / 59,64 (K 9,84 / 59,24) |

Eşli karşılaştırma (temel model / LoRA'nın daha iyi olduğu görev sayısı): `multi2`'de C@5 54 / 208,
C@10 23 / 162, NDCG@10 170 / 436; `multi3`'te C@5 17 / 109, C@10 15 / 167, NDCG@10 97 / 367 (hepsi
p < 0,0001). LiveMCPBench'te aynı ölçüler 4 / 6, 5 / 7, 27 / 34 (anlamlı değil; 94 görev).

Notlar:

- **Çok tool'luk setler nasıl kuruldu:** örneklenen her tool'a, kendi kaynağından ad ve açıklamasında
  en çok sözcük paylaşan 8 tool arasından (aynı sözcüklere sahip olanlar hariç) tohumlu seçimle 1–2
  ortak; model hepsini gerektiren tek görev yazıyor ya da `SKIP` diyor. Görevler adımları sayma
  eğiliminde ("önce … sonra …"), yani gerçek kullanıcı görevlerinden daha açık.
- **Sıralama artık benchmark'larla aynı yönde:** LoRA > head'ler > temel model > BM25, ve LoRA +
  head'ler LoRA'nın altında — ToolRet'teki sıra. Seçim seti olarak işe yarayan bunlar.
- **Tek tool'luk dev seti kolay.** Tek tool'luk, LLM'in tool metninden yazdığı istekleri gömme modelleri ilk 5'te
  %95–99 buluyor; ilk sıradaki hataların çoğu neredeyse aynı işi yapan komşu operasyonlar (iki
  modelin top-1 hataları simetrik: 76'ya 69). Bu yüzden set bir puanlık farkları sıralayamıyor;
  belirgin kötüleşmeyi yakalıyor (LoRA + head'ler: −1,3 / −3,3 NDCG@10; "durum" isteklerinde LoRA).
- **Uzunluk etkisi yok.** LoRA 768 token'lık tool metniyle eğitildi; açık, altın tool'un metin
  uzunluğuna göre değişmiyor (0–1.000, 1.000–2.500, 2.500+ karakter kovalarında aynı yön).
- **MCP-Zero'daki büyük kazanç istek kalıbına bağlı görünüyor:** o setin istekleri `server:` / `tool:`
  satırlarından oluşuyor; doğal dilde yazılmış tek tool'luk isteklerde (`dev_w3`) kazanç yok. Üç
  kontrol noktası arasından seçim bu kadar farkı açıklamaz; fark gerçek ama o kalıba özgü.
- `scripts/lora_train.py` yalnız en iyi adaptörü saklıyordu, bu yüzden mevcut LoRA yeni sette yeniden
  seçilemedi; `--keep-all` eklendi (her değerlendirilen adımın adaptörü kalır).
- İkinci aşamayla (`dev_w3`, ilk 20 + dokümantasyon, Qwen3-Reranker-8B) iki backbone ayırt edilemiyor:
  head'ler → reranker P@1 89,50 / Recall@5 99,60 / NDCG@10 95,51; LoRA → reranker 89,40 / 99,50 / 95,44
  (tek aşamaya göre top-1'de +5–6 puan; sorgu başına 0,5–1,3 sn).
- **Karar (kullanıcı, 2 Ekim):** LoRA varsayılan kalıyor (karar tek tool'luk sonuçlar görülerek
  verildi; çok tool'luk sonuçlar sonradan geldi ve kararı destekliyor). Model kartında ve README
  notunda kapsam yazılı: kazanç çok tool'luk görevlerde ve ToolRet / MCP-Zero'da, tek tool'luk
  isteklerde temel modelle başa baş.

Komutlar (GB10; sohbet modeli `--profile gen`, 8093, iş bitince durduruldu):

```bash
toolrank data gen-queries --data data/devcat --out data/dev_w3 --n 1000 --seed 0 \
  --exclude data/toolret --exclude data/livemcpbench_server --exclude data/mcp_zero_server
toolrank data gen-queries --data data/devcat --out data/dev_w3_sit --n 600 --seed 1 --styles situation \
  --exclude data/toolret --exclude data/livemcpbench_server --exclude data/mcp_zero_server
toolrank data gen-queries --data data/devcat --out data/dev_w3_multi2 --n 800 --seed 2 --tools-per-request 2 \
  --exclude data/toolret --exclude data/livemcpbench_server --exclude data/mcp_zero_server
toolrank data gen-queries --data data/devcat --out data/dev_w3_multi3 --n 500 --seed 3 --tools-per-request 3 \
  --gen-max-tokens 384 --exclude data/toolret --exclude data/livemcpbench_server --exclude data/mcp_zero_server
# data/devcat = Mac'teki data/w3'ün tools.jsonl'ı (sha256 f93cb95c…); her model için:
toolrank eval --data data/dev_w3_sit --scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora \
  --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst --ks 1,5,10 \
  --out results/dev_w3_sit_lora.json
```

## Komutlar

```bash
# GB10, ~/toolrank; FP8 servisi (compose profili lora-fp8, 8098) bf16'yla aynı birleştirilmiş ağırlıklardan
TOOLRANK_LORA=$HOME/toolrank/data/lora/qwen3-emb-lora-20k/merged \
  docker compose -f deploy/spark/compose.yaml --profile lora-fp8 up -d qwen3-emb-lora-fp8
FP8="--emb-url http://127.0.0.1:8098/v1 --emb-model qwen3-emb-lora-fp8 --truncate 8192 --emb-batch 128 \
  --tool-format documentation --query-format instruct_query --with-inst"
uv run toolrank eval --data data/toolret --scorer dense $FP8 --out results/lora_toolret_lora_fp8.json
uv run toolrank eval --data data/mcp_zero_server --scorer dense $FP8 --ks 1,5,10,20 --out results/lora_mcp_zero_server_lora_fp8.json
# v0.1 head'leri LoRA'nın üstünde (8097, bf16)
uv run toolrank eval --data data/livemcpbench_server --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
  --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --tool-format documentation \
  --query-format instruct_query --with-inst --out results/lora_livemcpbench_server_lora_v01heads.json
uv run python scripts/fp8_agreement.py --a http://127.0.0.1:8097/v1=qwen3-emb-lora --b http://127.0.0.1:8098/v1=qwen3-emb-lora-fp8 \
  --truncate 8192 --tool-format documentation --query-format instruct_query --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz
# README satırları
EMB_URL=http://127.0.0.1:8098/v1 EMB_MODEL=qwen3-emb-lora-fp8 TAG=lora_fp8 ROWS=qwen3emb bash scripts/readme_results.sh
EMB_URL=http://127.0.0.1:8097/v1 EMB_MODEL=qwen3-emb-lora TAG=lora ROWS=qwen3emb bash scripts/readme_results.sh
uv run python scripts/publish_backbone.py --model data/lora/qwen3-emb-lora-20k/merged      # kuru çalıştırma
# Mac
uv run python scripts/readme_table.py --write
```

## Ortam

GB10, NGC vLLM 26.01 (0.13), FP8 `--quantization fp8` yüklemede; ağırlıklar
`data/lora/qwen3-emb-lora-20k/merged` (diğer oturumun `scripts/lora_train.py` koşusu). Kod: `a0eca9a` sonrası.

## Sapmalar ve açıklamalar

- **FP8 servisi ilk denemede açılmadı:** 16 GB'lık tek ağırlık dosyası yüklenirken motor sessizce
  ölüp yeniden başladı (birleşik bellekte dört model açıktı). Ölçüm boyunca boşta duran bf16 LoRA
  sunucusu durduruldu (karşılaştırma vektörleri önbellekte), bitince geri açıldı; bge-reranker
  kullanıcının kararıyla durduruldu.
- **LiveMCPBench Recall@5'te LoRA bir sorgu geride** (52,06'ya karşı 53,03; 94 sorguda bir sorgu ≈ 1
  puan); NDCG@10'da önde (55,74'e karşı 53,95). Hiç görülmemiş MCP setlerindeki asıl kazanç ikinci
  aşamadan geliyor (LoRA + Qwen3-Reranker: 61,24 NDCG@10, `docs/reports/faz2-jev.md`).
- **MCP-Zero seçim seti:** LoRA'nın kontrol noktası MCP-Zero üzerinde seçildi; o sütun bağımsız değil
  (model kartında ve README notunda yazılı). backlog'daki "v0.2'den önce ayrı dev seti" maddesi bu
  ağırlıklar için de geçerli; seçimin etkisi küçük (üç kontrol noktası, 600. ve 625. adım neredeyse aynı).
- **Adlar önerildi, onaylanmadı:** `yasinyaman/toolrank-emb-8b`, `v0.2`, `toolrank-emb-v0.2`. Hepsi
  `build.py`'de tek yerde (giriş betiği, compose ve chart aynı değerleri taşıyor; bir test betiği
  `build`'e bağlıyor).
- Jev ikinci aşaması gerçek API'yle denenmedi (krediler 30 Eylül'de bitti); testler sahte istemciyle.

## Sonraki

- 19:00'dan sonra, kullanıcının onayıyla: GB10'dan `scripts/publish_backbone.py --upload`, sonra
  `BACKBONE_PUBLISHED = True` commit'i, sonra push.
- LoRA'yı büyütmek: `--keep-all` ile, seçim dört dev setinde (çok tool'luk kazanç korunurken tek
  tool'luk "durum" isteklerindeki açık kapanmalı), LiveMCPBench dokunulmadan.
