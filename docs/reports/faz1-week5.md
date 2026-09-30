# Faz 1 — Hafta 5 raporu (30 Eyl 2026)

Takvimde 30 Kasım – 4 Aralık; erken yapıldı.

## Sonuç (tek cümle)

Hafta 5'in üç maddesi bitti:

- `toolrank finetune`, Faz 0'ın head eğitimini tek komutla ve aynı sayılarla yeniden üretiyor; epoch'u
  eğitim çiftlerinden değil ayrı bir dev setinden seçiyor. Faz 0'ın tuzağını da kendisi yakalıyor:
  mined negatiflerle doğrulama recall'u 89.45'ten 96.85'e çıkarken dev NDCG@10 87.21'den 78.65'e
  düştü ve komut başlangıç head'lerini geri verdi.
- toolrank artık LangGraph (langgraph-bigtool), LlamaIndex ve LiteLLM proxy'sinde çalışıyor.
  `data/w3`'te (1.862 tool) bigtool ve LlamaIndex iki görevde de doğru tool'u buldu. LiteLLM
  filtresi 120 tool'luk istekleri üç API biçiminde de 3 function tool'a indirdi.
- README'de her hücresi protokole göre denetlenen bir sonuç tablosu var. v0.1 head'leri ToolRet'te
  54.03 (cat-macro 47.13), LiveMCPBench'te 53.03, MCP-Zero'da 79.87 veriyor.

## 1. `toolrank finetune`

Embed, eğitim ve resmi değerlendirme tek komutta.

- **Veri.**
  - Pozitifi olmayan çiftler atılıyor. İsteği dev ya da eval setlerinden birindeki bir sorguyla
    aynı olan çiftler de atılıyor; boşluk ve büyük/küçük harf farkı sızıntıyı gizlemiyor. Sayılar
    kaynak başına raporlanıyor.
  - Talimatı olmayan çiftlere servis talimatı ekleniyor, çünkü head'ler talimatla servis ediliyor
    (`--keep-bare` bunu kapatıyor).
- **Bir kez embed.** Eğitim, doğrulama, dev ve eval metinlerinden önbellekte olmayanlar gömülüyor.
  `--embed-only` burada duruyor: torch gerektirmiyor ve yarıda kalırsa kaldığı yerden sürüyor.
- **Eğitim.** Varsayılanlar Faz 0'ın ayarı: skip head (x + MLP(x), başlangıçta kimlik), genişlik
  1536, lr 1e-5, batch 512, 5 epoch, yalnız in-batch negatif.
- **Seçim.**
  - Her epoch'ta dev seti, önbellekteki vektörlerle ve `toolrank eval`'in hesabıyla puanlanıyor
    (`--select ndcg10 | ndcg10-cat`). `--curve` eval setlerini de puanlıyor; bu sayılar yalnız
    raporlanıyor, seçime girmiyor.
  - Başlangıç (epoch 0) da yarışıyor. Sonraki bir epoch ancak kesin olarak daha iyiyse seçiliyor;
    hiçbiri geçemezse çıktı başlangıç head'leri oluyor.
- **Başlangıç noktası.** `--init-ckpt` bir `.pt`, paketlenmiş bir `.npz` ya da `default`
  (yayımlanan head'ler) alıyor. `.npz`'nin fp16 ağırlıkları fp32'ye açılıyor; gizli boyut
  vektörlerle karşılaştırılıyor.
- **Çıktı.**
  - `.pt` (istenirse `--npz` de) servis ayarlarını taşıyor: backbone, formatlar, truncate 8192,
    talimat. `toolrank eval` ve `serve` bunları bayrak gerekmeden doğru kullanıyor.
  - cfg'de ayrıca `selected_on` (set, sorgu sayısı, metrik, değer, epoch) ve `trained_from` var.
    `trained_from` dosya adı ve sha256; yerel yol yazılmıyor, çünkü cfg paketlenen dosyayla
    dağıtılıyor.
  - Kaydedilen dosyayla dev ve eval setlerinde resmi `toolrank eval` koşuyor. Rapor
    (`results/finetune_<ad>.json`) şunları içeriyor: eğri, seçilen epoch, atılan çiftler, token,
    sha256'lar ve eğri ile resmi sayı arasındaki fark.
- **Reddettikleri.**
  - Bir eval setiyle aynı olan ya da onunla sorgu paylaşan dev seti (`mcp_zero` ile
    `mcp_zero_server` gibi). Seçimde kullanılan setin sayısı raporlanamaz.
  - Kategorisi olmayan bir dev setinde `ndcg10-cat`.
  - `--init-ckpt` ile birlikte `--width`, `--depth` ya da `--no-skip`.
  - 300 sorgudan küçük bir dev seti için uyarı veriyor.
- **Kaldırılanlar.** `train/embed_pairs.py` ve `train/finetune_heads.py`. İki betik bölmeyi
  birbirinden farklı yapıyordu.
- **Testler.**
  - CI torch'suz koşuyor. Orada sızıntı filtresi, gömülecek metinler, reddetmeler ve `.npz`
    okumanın numpy tarafı sınanıyor. `--embed-only` ikinci koşuda endpoint'e hiçbir şey
    göndermiyor.
  - Mac'te torch'la sınananlar: dev eğrisinin `run_eval` ile birebir aynı çıkması, seçim kuralı,
    `.npz`'den devam ve uçtan uca komut (sentetik set, sahte endpoint).

## 2. Fine-tune koşuları (GB10)

Ortak ayarlar:

- ToolRet-train'den 60.000 eğitim ve 2.000 doğrulama çifti (seed 0).
- Dev: MCP-Zero `_server` (2.792 sorgu, NDCG@10).
- `--curve` ile ToolRet ve LiveMCPBench `_server`. Hepsi w/ inst.
- 776 çift, istekleri ToolRet sorgularıyla aynı olduğu için atıldı (MCP setlerinde 0).
- 131.599 metnin hepsi önbellekten geldi; encoder'a 0 token gitti.

Epoch başına sonuçlar aşağıda. Val sütunu doğrulama çiftlerinde recall@10; diğer sütunlar NDCG@10.
Kalın satırlar dev'in seçtiği epoch'lar:

| Negatif | Epoch | Val | Dev (MCP-Zero) | ToolRet micro / cat-macro | LiveMCPBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| — | **0 (başlangıç)** | 89.45 | 87.21 | 51.11 / 46.55 | 53.74 |
| in-batch | 1 | 92.15 | 87.54 | 51.27 / 45.79 | 54.92 |
| | **2** | 93.65 | **88.50** | 51.75 / 46.70 | 55.64 |
| | 3 | 94.30 | 88.35 | 52.18 / 47.04 | 53.58 |
| | 4 | 94.75 | 88.25 | 52.54 / 47.20 | 55.26 |
| | 5 | 94.65 | 88.33 | 52.58 / 47.07 | 54.86 |
| 15 mined | 1 | 95.35 | 83.27 | 46.32 / 38.52 | 52.10 |
| | 2 | 96.10 | 78.65 | 45.01 / 35.79 | 51.76 |
| | 3 | 96.00 | 78.64 | 45.08 / 35.39 | 50.68 |
| | 4 | 96.85 | 79.16 | 45.81 / 36.19 | 51.40 |
| | 5 | 96.85 | 79.43 | 46.10 / 36.38 | 51.10 |

- **Faz 0 ile eşlik.** Faz 0'ın kuralıyla, yani val recall'a göre seçilseydi, iki koşu da epoch 4'ü
  seçerdi: 52.54 / 47.20 ve 45.81 / 36.19. İkisi de `faz0-week3.md`'deki satırlarla birebir aynı.
  Yeni kod eski sonuçları değiştirmedi.
- **In-batch negatif.**
  - Dev epoch 2'yi seçti (88.50).
  - Val recall'un seçeceği epoch 4'e göre ToolRet'te 0.8 / 0.5 puan geride, LiveMCPBench'te 0.4
    puan önde. Epoch 2'den sonra dev eğrisi düz.
  - ToolRet'in epoch'larla artması beklenen bir şey: eğitim çiftleri ToolRet'in kendi dağılımından.
- **Mined negatif.**
  - Val recall 89.45'ten 96.85'e çıktı. Aynı sürede dev 87.21'den 78.65'e, ToolRet cat-macro
    46.55'ten 35.39'a düştü.
  - Dev hiçbir epoch'ta başlangıcı geçemedi. Komut epoch 0'ı, yani eğitilmemiş skip head'leri
    (dense taban çizgisini) geri verdi.
  - Faz 0'da bu tuzak elle fark edilmişti; artık komut kendisi yakalıyor.
- **Yayımlanan head'lerden devam** (`--init-ckpt default --epochs 1 --n-train 5000`).
  - Epoch 0'ın dev sayısı 88.53, yani v0.1 head'lerinin kendi sayısı.
  - Resmi eval ToolRet'te 54.03 / 47.13 verdi; bunlar da v0.1'in sayıları.
  - Tek epoch dev'i 88.44'e düşürdü, bu yüzden epoch 0 geri verildi.
  - cfg'de `trained_from` = `toolrank-heads-qwen3-emb-8b-v0.1.npz` ve `init_sha256` = `f3c10125…`.
- **Eğri = resmi sayı.** Seçilen epoch'un eğrideki değeriyle kaydedilen dosyanın resmi
  `toolrank eval` sonucu arasındaki en büyük fark 0.0002 (ToolRet cat-macro). MCP setlerinde fark 0.
- **Süre.** 60 bin çiftte bir epoch, üç setin eğrisi dahil, ~18 sn sürüyor (15 mined negatifle
  ~22 sn). Head'ler GB10'un GPU'sunda eğitiliyor.
- **Yayımlanmadı.** Bu head'lerin hiçbiri yayımlanmadı. Dev seti olarak kullanılan MCP-Zero aynı
  zamanda raporladığımız bir benchmark; v0.2 head'lerinden önce ayrı bir dev seti gerekiyor
  (`toolrank data gen-queries`, backlog'da).

## 3. Framework adapter'ları

Üçü de `toolrank serve`'e `ToolrankClient` üzerinden bağlanıyor.

- **LangGraph** (`integrations/langgraph.py`, `toolrank[langgraph]`):
  - `Toolbox.registry()` kataloğun her tool'unu bir LangChain aracı yapıyor. Şema tool'un
    `inputSchema`'sı; çağrıyı toolrank çalıştırıyor (`/v1/call`).
  - `retrieve_tools(query) -> list[str]`, langgraph-bigtool'un `retrieve_tools_function`'ı. Yalnız
    kayıt anahtarlarını döndürüyor, çünkü bigtool bilinmeyen bir anahtarda hata veriyor. Kaç tool
    döneceğini uyarlanabilir K belirliyor. Fonksiyonun docstring'i modelin gördüğü açıklama.
  - Araçlar yalnız modelin argümanlarını alıyor. LangChain `config` ve `run_manager` adlı
    parametrelere kendi değerlerini koyuyor; araçların böyle parametresi olsaydı bu adlı tool
    argümanları kaybolurdu.
  - Başarısız, reddedilen ya da onaylanmayan bir çağrı, modelin okuyacağı bir hata sonucuna
    dönüşüyor.
  - Kullanım günlüğünde oturum LangGraph thread'i: `langgraph:<thread_id>`.
  - bigtool'un PyPI'daki 0.0.3 sürümü langgraph 1.x'te çalışmıyor, `langgraph<1` gerekiyor. Bizim
    modül yalnız langchain-core istiyor.
- **LlamaIndex** (`integrations/llamaindex.py`, `toolrank[llamaindex]`): `ToolrankToolRetriever`,
  bir `ObjectRetriever`.
  - LlamaIndex 0.14 ajanları her adımda kullanıcı mesajıyla, her çağrıdan önce de tool'un adıyla
    arıyor. Retriever bir sorgunun sonucunu 300 sn tutuyor, böylece aynı arama tekrarlanmıyor.
    Daha önce verdiği bir tool'un adı gelince aramadan o tool'u döndürüyor.
  - Açıklamalar OpenAI'ın kabul ettiği 1.024 karaktere kesiliyor. `fn_schema`, tool'un
    `inputSchema`'sını veren bir pydantic modeli.
- **LiteLLM** (`integrations/litellm.py`): proxy'nin ayarında
  `callbacks: toolrank.integrations.litellm.tool_filter`.
  - İstek öncesi kancası, isteğin kendi function tool'larını `/v1/rank` ile sıralıyor ve
    uyarlanabilir K'nın tuttuklarını bırakıyor. Konuşmada çağrılmış tool'lar, `tool_choice`'un
    adlandırdığı tool ve sağlayıcı tool'ları (web search gibi) her zaman kalıyor.
  - Chat Completions, Responses ve Anthropic Messages isteklerini anlıyor. LiteLLM'in kendi
    `mcp_semantic_tool_filter`'ı yalnız Chat Completions'ta ve sabit top-K ile çalışıyor.
  - Şu isteklere dokunmuyor: 20 ya da daha az function tool'u olan, sağlayıcının kendi tool
    aramasını kullanan (`defer_loading`) ve kullanıcı metni olmayan. toolrank hata verirse ya da
    2 sn içinde cevap vermezse istek filtresiz gidiyor. İsteği yerinde değiştirmiyor.
  - Proxy yalnız sınıfın kendi gövdesinde tanımlı kancayı çağırıyor, kalıtılmış kancayı atlıyor.
    Kanca bu yüzden sınıfın kendisinde tanımlı.
  - Paket ekstrası yok: litellm `openai<3` istiyor ve `[dev]`'deki `openai>=3.22` ile çakışıyor.
    Filtre proxy'nin kendi ortamında çalışıyor; litellm yalnız proxy filtreyi yüklerken import
    ediliyor.
  - LiteLLM'in MCP ağ geçidinin kendi eklediği tool'lar bu kancadan sonra ekleniyor. Onlar için
    toolrank ağ geçidinin arkasına MCP sunucusu olarak konuyor. `examples/litellm/config.yaml` iki
    yolu da gösteriyor.
- **Testler.** LangGraph ve LlamaIndex testleri gerçek sınıflarla ve süreç içinde çalışan bir
  toolrank'la koşuyor; LlamaIndex'te bir `FunctionAgent` sahte bir LLM'le baştan sona çalışıyor.
  litellm venv'de olmadığı için filtrenin mantığı litellm'siz sınanıyor, LiteLLM sınıfı da sahte
  bir `CustomLogger` ile. Gerçek litellm 1.103.1 filtreyi yol adından yükledi ve
  `ProxyLogging.pre_call_hook` üzerinden 25 tool'u 2'ye indirdi.

### Uçtan uca (`scripts/frameworks_e2e.py`, `data/w3`, 1.862 tool)

Modeller betikli: önce arıyor, bulunan tool'lar arasında görevin tool'u varsa onu çağırıyor, sonra
cevap veriyor. Hiçbir model API'si çağrılmadı; hiçbir tool çağrısı makineden çıkmadı (OpenAPI
kaynakları yerel taklide yönlendirildi).

| Framework | Görev | Bulunan tool'lar | Çağrı | Süre |
| --- | --- | --- | --- | ---: |
| langgraph-bigtool (sync) | Tokyo'da saat kaç? | `time__get_current_time`, `time__convert_time` | `time/get_current_time` | 0.49 sn |
| langgraph-bigtool (async) | Köpek etiketli hayvanlar | `swagger-petstore__findPets`, `swagger-petstore__addPet` | `findPets` (Rex, Ace) | 0.05 sn |
| LlamaIndex `FunctionAgent` | Tokyo'da saat kaç? | aynı iki tool | `time/get_current_time` | 0.09 sn |
| LlamaIndex `FunctionAgent` | Köpek etiketli hayvanlar | aynı iki tool | `findPets` (Rex, Ace) | 0.01 sn |

LlamaIndex ajanı her görevde iki adım attı ama bir kez aradı; ikinci adım önbellekten geldi.

LiteLLM proxy (litellm 1.103.1) önünde bir taklit model API'si vardı; taklit, her isteğin
getirdiği tool'ları kaydetti. İsteklerde 120 katalog tool'u (küçük sunucuların hepsi, sonra
sırayla Stripe ve GitHub) ve Responses ile Messages'ta bir sağlayıcı tool'u vardı. Geçmişte bir
`stripe__GetAccount` çağrısı vardı:

| Biçim | Görev | Gönderilen | Modele giden | Süre |
| --- | --- | ---: | --- | ---: |
| Chat Completions | Tokyo | 120 | `time__get_current_time`, `time__convert_time`, `stripe__GetAccount` | 11.49 sn |
| Chat Completions | Köpekler | 120 | `swagger-petstore__findPets`, `swagger-petstore__addPet`, `stripe__GetAccount` | 0.06 sn |
| Responses | Tokyo | 121 | `web_search` ve Chat'teki aynı üç tool | 0.41 sn |
| Responses | Köpekler | 121 | `web_search` ve Chat'teki aynı üç tool | 0.04 sn |
| Anthropic Messages | Tokyo | 121 | `web_search` ve Chat'teki aynı üç tool | 0.43 sn |
| Anthropic Messages | Köpekler | 121 | `web_search` ve Chat'teki aynı üç tool | 0.03 sn |

- Her biçimde modele 120 yerine 3 function tool gitti. Görevin tool'u hep içindeydi; geçmişte
  çağrılan tool ve sağlayıcı tool'u korundu.
- İlk istek 11.5 sn sürdü: 120 tool metni ilk kez görüldü ve GB10'da gömüldü. Bu koşuda zaman
  aşımı bu yüzden uzun tutuldu. Varsayılan 2 sn'de bu istek filtresiz giderdi, gömme arka planda
  sürerdi. Sonraki istekler önbellekten geldi.
- MCP ağ geçidi: LiteLLM, toolrank'ın iki aracını `toolrank-search_tools` ve `toolrank-call_tool`
  adıyla sundu. Arama `time/get_current_time`'ı ilk sırada buldu ve çağrı çalıştı.
- Kullanım günlüğü: 5 çağrının 5'i de `search_id` ile aramasına bağlandı. Oturumlar: bigtool'da
  LangGraph thread'leri (`langgraph:e2e-bigtool-0`, `-1`), LlamaIndex'te retriever'ın oturumu, ağ
  geçidinde LiteLLM'in MCP oturumu.

## 4. README sonuç tablosu

- `docs/results.toml` tablonun satırlarını tanımlıyor. Ölçülen her satır, her benchmark için bir
  `toolrank eval` raporunu adlandırıyor; raporlar `docs/results/`'ta, repoda. Yayımlanmış sayılar
  (ToolRet makalesi, StackOne) yalnız ait oldukları sütunda, ToolRet cat-macro'da duruyor.
- `scripts/readme_table.py --write` tabloyu README'de iki işaretin arasına yazıyor. Yazmadan önce
  her raporu protokole göre denetliyor:
  - benchmark ve sorgu sayısı;
  - `--limit`, `--tasks` ve `--instruction` kullanılmamış olmalı;
  - top-100, hibrit değil;
  - satırın talimat ayarı ve, satır bir head dosyası adlandırıyorsa, o dosyanın sha256'sı.

  Bir satırın raporları tool ve sorgu formatında, embedding modelinde, kırpmada ve head dosyasında
  aynı olmalı. Bir test README'deki tabloyu raporlardan üretilenle karşılaştırıyor, böylece CI bayat
  bir tabloyu yakalıyor.
- `toolrank eval` artık kullandığı head dosyasının sha256'sını rapora yazıyor.
  `toolrank data server-names`, Faz 0'ın `_server` setlerini izlenen koddan üretiyor; GB10'da iki
  set de Faz 0'dakilerle bayt bayt aynı çıktı.
- 12 rapor güncel kodla `scripts/readme_results.sh` ile yeniden üretildi. Hepsi önbellekten geldi
  (encoder'a 0 token); sayılar Faz 0 ve Hafta 2'dekilerle aynı.

| Retriever | ToolRet NDCG@10 | cat-macro | LiveMCPBench Recall@5 | MCP-Zero top-1 |
| --- | ---: | ---: | ---: | ---: |
| BM25, talimatsız | 29.01 | 22.24 | 31.68 | **80.44** |
| BM25, talimatlı | 39.27 | 36.41 | 22.92 | 45.63 |
| Qwen3-Embedding-8B | 51.11 | 46.54 | 50.82 | 78.19 |
| Qwen3-Embedding-8B + head'ler v0.1 | **54.03** | 47.13 | **53.03** | 79.87 |
| NV-Embed-v1 (ToolRet makalesi) | — | 42.71 | — | — |
| gte-Qwen2-1.5B-instruct (ToolRet makalesi) | — | 45.96 | — | — |
| StackOne v2 (StackOne) | — | **54.40** | — | — |

- **Yeni hücreler.** Talimatlı BM25, MCP setlerinde ilk kez ölçüldü. Genel talimat BM25'i düşürüyor:
  LiveMCPBench 31.68'den 22.92'ye, MCP-Zero 80.44'ten 45.63'e.
- **Olduğu gibi gösterilenler.** MCP-Zero'da top-1'de talimatsız BM25 önde (80.44'e 79.87). ToolRet
  cat-macro'da StackOne v2 önde (54.40'a 47.13); head'lerimiz Faz 0 kapısının (50) da altında.
  İkisi de tabloda ve notlarda açıkça yazıyor.

## Komutlar

```bash
# GB10, ~/toolrank
toolrank finetune --data data/toolret_train/pairs.jsonl --n-train 60000 --n-val 2000 \
  --dev data/mcp_zero_server --eval data/toolret --eval data/livemcpbench_server --curve \
  --out data/heads/w5_60k.pt --npz dist/heads/w5_60k.npz --name w5_60k
toolrank finetune --data data/toolret_train/pairs.jsonl --n-train 60000 --n-val 2000 --neg 15 \
  --dev data/mcp_zero_server --eval data/toolret --eval data/livemcpbench_server --curve \
  --out data/heads/w5_60k_neg15.pt --name w5_60k_neg15
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz toolrank finetune \
  --data data/toolret_train/pairs.jsonl --init-ckpt default --epochs 1 --n-train 5000 \
  --dev data/mcp_zero_server --eval data/toolret --eval data/livemcpbench_server \
  --out data/heads/w5_init.pt --name w5_init
# GB10, ~/toolrank: README tablosunun 12 koşusu; ardından Mac'te raporları çekip tabloyu yaz
PYTHONUNBUFFERED=1 nohup bash scripts/readme_results.sh > data/logs/readme_results.log 2>&1 &
scp 'gb10:toolrank/results/readme_*.json' docs/results/ && uv run python scripts/readme_table.py --write
# GB10: _server setleri izlenen koddan, Faz 0'dakilerle karşılaştırma (sha256)
toolrank data server-names data/livemcpbench --out /tmp/v0_livemcpbench_server
# Mac, repo kökü: framework'ler uçtan uca (bigtool kendi ortamında, LiteLLM uvx ile)
# $GB10: GB10'un Tailscale adresi (depoda tutulmuyor). Embedding cache'i URL'in yazımıyla anahtarlı: hep aynı adres.
uv run python scripts/frameworks_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e.json
```

## Ortam

- GB10: Qwen3-Embedding-8B, NGC imajındaki vLLM ile 8091'de, `--max-model-len 8192`; embedding
  önbelleği sıcak. Head'ler torch 2.14.0+cu130 ile GPU'da.
- Mac: Python 3.13; langchain-core 1.6.6, langgraph 0.6.11 ile langgraph-bigtool 0.0.3,
  llama-index-core 0.14.25, litellm 1.103.1 (ayrı ortamda).
- Commit'ler: `3e5fb39` … `ca24258`, ardından bu rapor.

## Sapmalar ve açıklamalar

- **Dev seti bir benchmark.** Seçim MCP-Zero `_server` üzerinde yapıldı. Bu head'lerin MCP-Zero
  sayısı bu yüzden artık bağımsız bir ölçüm değil; LiveMCPBench ve ToolRet seçime girmedi.
  Ayrı dev seti backlog'da; v0.2 head'lerinden önce şart.
- **Plandan sapmalar.**
  - `litellm` ekstrası yok: litellm `openai<3` istiyor, `[dev]` ise `openai>=3.22`. Filtre proxy'nin
    kendi ortamına kurulan toolrank'la çalışıyor.
  - LlamaIndex modülü import edilirken llama-index-core istiyor, çünkü retriever bir LlamaIndex
    sınıfı. LangGraph ve LiteLLM modülleri framework'ü yalnız kullanılınca yüklüyor.

## Sonraki hafta

Faz 1 Hafta 6, dokümantasyon ve paketleme:

- docs sitesi (mkdocs): kavramlar, hızlı başlangıç, benchmark, mimari, katkı rehberi;
- PyPI yayını, Docker imajı (vLLM backbone + toolrank) ve `docker compose` örneği;
- lisans ve üçüncü taraf bildirimleri, CODE_OF_CONDUCT, issue şablonları.

Head'lerin yüklenmesi (Hafta 2'nin 4. kutusu) hâlâ açık: `hf auth login` gerekiyor.

