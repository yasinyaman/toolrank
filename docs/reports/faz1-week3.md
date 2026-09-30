# Faz 1 — Hafta 3 raporu (30 Eyl 2026)

Takvimde 16–20 Kasım; erken yapıldı.

## Sonuç (tek cümle)

`toolrank serve` bir ingest dizinini ajanlara iki tool'luk tek bir MCP sunucusu olarak sunuyor:
`search_tools` ve `call_tool`. Sunucu HTTP'den ya da stdio'dan çalışıyor; aynı portta bir REST API
ve OpenAPI şeması da var. Her arama ve çağrı Faz 2 için kullanım günlüğüne yazılıyor.

Mac'te uçtan uca denendi: 1.862 tool, 5 kaynak, embedding'ler GB10'dan Tailscale üzerinden.

- Sıcak sorguyla arama MCP üzerinden 5 ms (p50), proxy'nin çağrı ek yükü ~2 ms.
- Boş cache'le ilk açılış 154 sn sürüyor; cache doluyken 0.8 sn.
- Test sırasında bulunan iki hata düzeltildi: servis zaman aşımı indeks kurulumunu da kesiyordu;
  ingest'in ısıttığı cache'i serve görmüyordu.

## Ne yapıldı

- **Retriever (`retriever.py`, çekirdek).** Değişmez bir durum tutuyor: tool'lar, id haritası ve
  indekslenmiş scorer. `tools.jsonl` değişince (mtime + boyut, her aramada bakılıyor) taze bir
  scorer kuruluyor ve durum tek atamayla değiştiriliyor. Aramalar kilitsiz, thread'lerde
  çalışıyor.
  - İlk indeks arka planda kuruluyor. El sıkışma ve `list_tools` hemen cevaplanıyor; arama indeksi
    30 sn bekliyor, sonra "hâlâ kuruluyor, tekrar dene" diyor.
  - Scorer'lara gerekenler eklendi: hibritte `rank_pairs` (paylaşılan durum yok), dense'te
    `score_tools`. SQLite cache thread'ler arası kilitli, FAISS'te sorgu başına `efSearch`,
    pgvector bağlantısı kilitli.
- **MCP proxy (`adapters/mcp_proxy.py`).** SDK'nın alt düzey `Server`'ı üzerine kurulu ve iki tool
  sunuyor.
  - `search_tools(query, k?)` şunları döndürüyor: tool adı, sunucu, skor, açıklama, `inputSchema`
    (ilk 3 tool'da tam, sonrakilerde kısaltılmış) ve bir `search_id`.
  - `call_tool(name, arguments, search_id?)` çağrıyı tool'un MCP sunucusuna ya da OpenAPI işlemine
    iletiyor. MCP sonucu olduğu gibi dönüyor. Hatalı çağrıda tool'un tam şeması da hata metnine
    ekleniyor.
  - Taşıma: streamable HTTP `/mcp` ya da `--stdio`.
- **Arka uçlar (`adapters/backends.py`).**
  - MCP: her sunucu için bir `mcp.Client`. Oturum ilk çağrıda açılıyor, çökünce yeniden açılıyor.
    Yarım kalmış bir çağrı asla tekrar gönderilmiyor. Zaman aşımı, protokol hatası ve komut
    bulunamadı durumları istisna olarak değil, `isError` sonucu olarak dönüyor.
  - OpenAPI: `httpx2` ile gerçek HTTP.
    - Path parametreleri kodlanıyor. Query `form`/`deepObject` stiliyle gidiyor (ingest artık
      stili kaydediyor). Body JSON ya da köşeli parantezli form (Stripe).
    - Yalnız GET/HEAD gidiyor; yazma işlemleri `--allow-write` ister.
    - Config'teki header'lar yalnız config'teki `base_url`'e gidiyor. `${ENV}` değerleri ortamdan
      açılıyor.
    - Yönlendirme izlenmiyor; cevap 25 bin karakterde kesiliyor.
  - Sunucunun yaşam döngüsü arka uçları sahipleniyor: kapanınca hiçbir stdio alt süreci kalmıyor.
- **REST (`adapters/rest.py`).** `/mcp` ile aynı Starlette uygulamasında çalışıyor.
  - `POST /v1/search`.
  - `POST /v1/rank`: çağıranın verdiği en fazla 200 MCP tool'unu ya da katalog id'lerini kosinüsle
    sıralıyor. Her satır istekteki `index`'iyle dönüyor.
  - `GET /v1/tools[?server=]`, `GET /v1/tools/{id}`, `GET /openapi.json` (elle yazılmış 3.1),
    `GET /healthz` (hazır olana dek 503).
  - Aynı katmanda bearer token (`/mcp` ve `/v1`), Host denetimi ve 1 MiB gövde sınırı var. Sınır
    okunurken de uygulanıyor, yani chunked gövde de sınırı aşamıyor. Loopback dışı `--host`
    API anahtarı istiyor.
- **Kullanım günlüğü (`usage.py`, şema v1).** Kayıtlar günlük JSONL dosyalarına gidiyor; her olay
  `O_APPEND` üzerinde tek bir `os.write`, böylece stdio ve HTTP sunucuları aynı dizini paylaşabiliyor.
  - `search` olayı: ilk 20 [tool, skor], gösterilen sayı, kural, süre, scorer, head sha'sı, katalog
    sha'sı, embedding-cache anahtarı.
  - `call` olayı: tool, tür, sonuç (`ok | tool_error | protocol_error | timeout | refused |
    unknown_tool`), HTTP durumu, süre.
  - İstek ve argümanlar kurulum başına anahtarla (0600) HMAC'leniyor. Metin yalnız `--log-text`
    ile yazılıyor.
  - Çağrı bir aramaya bağlanıyor. Bağlama sırası: ajan `search_id` verdiyse o arama; yoksa
    oturumun son araması; o da yoksa aynı arayüzden (`via`) son arama.
- **CLI.** `toolrank serve --data DIR [--config F] [--server N=T] [--stdio] [--host] [--port]
  [--api-key] [--allowed-host] [--allow-write] [--usage-log | --no-usage-log] [--log-text]
  [--timeout]` + `search`'ün arama bayrakları. Yollar mutlak yapılıyor (Claude Desktop sunucuları
  `/`'da başlatıyor). README'de Claude Desktop örneği var.

Testler hermetik: fixture MCP sunucusu (stdio; yavaş, çöken ve pid dönen tool'larla),
`httpx2.MockTransport` ve Starlette `TestClient`. Mac'te 102 test geçiyor, 7'si atlanıyor (torch
ve Postgres).

## Uçtan uca doğrulama

Kurulum:

- Mac (`scripts/serve_e2e.py`). Embedding: GB10'daki Qwen3-Embedding-8B (8091), Tailscale
  üzerinden. Head'ler fp16 `.npz`.
- Katalog `data/w3`: Hafta 1'in dizini, `everything-http` çıkarılmış; time, everything,
  swagger-petstore, stripe, github olmak üzere 1.862 tool.
- OpenAPI kaynakları sorgu stilleri için yeniden ingest edildi (GitHub 332, Stripe 273 satır yalnız
  `doc.http`'de değişti; metin aynı, yeniden gömme yok).
- Petstore spec'inin sunucusu başka bir API; betik yerine yerel bir taklit ayağa kaldırıp config'i
  ona yönlendiriyor (`X-Api-Key: ${PETSTORE_KEY}` ile). Böylece hiçbir tool çağrısı makineden
  çıkmıyor. Stripe ve GitHub yalnız aranıyor.

### İlk açılış

| Durum | HTTP portu | MCP `list_tools` | İndeks hazır | İlk arama |
| --- | ---: | ---: | ---: | --- |
| Soğuk: cache ve indeks boş | 0.61 sn | 0.88 sn | 154.1 sn (1.862 tool gömüldü) | 30 sn bekledi, "hâlâ kuruluyor" hatası |
| Cache dolu, indeks yok | 0.67 sn | 1.07 sn | 0.8 sn (1.862 projeksiyon) | 1.18 sn'de sonuç |
| Cache ve indeks dolu (stdio) | — | 1.3 sn (süreç başlangıcı dahil) | 0.0 sn | 1.57 sn'de sonuç |

- İndeks hazır olunca `search_tools`'un açıklaması "the 1862 tools of 5 servers and APIs" oluyor.
- Soğuk açılışın tamamı embedding (GB10'da, ağ üzerinden).
- `ingest --emb-url` bu işi önceden yapıyor. Artık serve ile aynı cache'i (`DIR/cache`) ve aynı
  model/kırpmayı kullandığı için sıcak açılış ~1 sn. Doğrulama: serve'ün doldurduğu cache'te
  ingest "embedded 0 … 1862 already cached" diyor.

### Arama

Adaptif K (margin 0.2, max 10). Süreler istemci tarafında, MCP HTTP çağrısı dahil.

| İstek | Dönen tool'lar | Soğuk (sorgu gömme) | Sıcak |
| --- | --- | ---: | ---: |
| what time is it in Tokyo right now | `time/get_current_time`, `time/convert_time` | 192 ms | 6.6 ms |
| echo back this message | `everything/echo` | 131 ms | 5.7 ms |
| list the pets tagged dog | `swagger-petstore/findPets`, `addPet` | 140 ms | 6.7 ms |
| open an issue in my repository about the login bug | 10 tool; ilki `github/issues/create` | 139 ms | 9.1 ms |
| refund the last payment of this customer | 10 Stripe iade işlemi (`PostRefunds` 2.) | 194 ms | 8.0 ms |
| cancel the subscription at the end of the billing period | 10 Stripe abonelik işlemi | 155 ms | 11.2 ms |

- Aynı sorgu 20 kez: p50 5.2 ms, p90 6.5 ms.
- REST `/v1/search` (sıcak): 3.6 ms duvar saati, 2.4 ms sunucu içi.
- Soğuk sürenin ~130–190 ms'si sorgunun Tailscale üzerinden gömülmesi.
- Net isteklerde 1–2 tool dönüyor, geniş isteklerde 10 (üst sınır).
- Son istekte doğru işlem (`cancel_at_period_end` ile abonelik güncelleme,
  `PostSubscriptionsSubscriptionExposedId`) 10. sırada; hemen iptal (`DELETE`) 2. sırada.

### Çağrılar

| Çağrı | Sonuç | ms |
| --- | --- | ---: |
| `time/get_current_time` {timezone: Asia/Tokyo} | doğru saat (ilk çağrı: `uvx` süreci başlıyor) | 689 |
| `everything/echo` | "Echo: hello from toolrank" (ilk çağrı: `npx`) | 1.459 |
| `swagger-petstore/findPets` {tags: [dog, cat], limit: 2} | HTTP 200; `tags=dog&tags=cat&limit=2`, API anahtarı ulaştı | 35 |
| `swagger-petstore/find pet by id` {id: 404} | `isError`: HTTP 404 + tool'un tam şeması | 4.9 |
| `stripe/PostRefunds` | reddedildi: POST, `--allow-write` yok (istek çıkmadı) | 3.0 |
| `time/get_current_time` {tz: …} | `isError`: sunucunun doğrulama hatası + tam şema | 3.9 |
| `nope/missing` | `isError`: "önce search_tools" | 1.5 |

Kararlı durumda 20'şer çağrı (proxy + arka uç):

| Çağrı | p50 | p90 |
| --- | ---: | ---: |
| `time/get_current_time` (stdio) | 2.6 ms | 2.9 ms |
| `everything/echo` (stdio) | 1.8 ms | 2.4 ms |
| `swagger-petstore/findPets` (yerel HTTP) | 2.3 ms | 2.5 ms |

stdio'da (Claude Desktop gibi, `cwd=/`): `time/convert_time` 371 ms (ilk çağrı), `everything/echo`
662 ms (ilk çağrı).

Kapanış: HTTP sunucusu SIGINT'le, stdio istemcisi bağlantıyı kapatarak durdu. İkisinden sonra da
hiçbir arka uç süreci kalmadı (`mcp-server-time`, `server-everything` için `pgrep`).

### REST

| Kontrol | Sonuç |
| --- | --- |
| token yok / yanlış Host | 401 / 421 |
| `/v1/rank`, istemcinin 3 tool'u, "email my boss that I will be late" | send_email 0.566 > create_event 0.290 > get_weather 0.115 |
| `/v1/rank`, katalog id'leri, "what time is it in London" | get_current_time 0.598 > convert_time 0.489 > echo 0.002 |
| bilinmeyen id | 404 |
| `/v1/tools?server=time`, `/v1/tools` | 2, 1.862 |
| `/v1/tools/swagger-petstore%2Ffind%20pet%20by%20id` (boşluklu ad) | 200, `openapi`, `/pets/{id}` |
| `/openapi.json`, `/healthz` (token'sız) | 200 (3.1.0, 5 yol), 200 |

### Kullanım günlüğü

Soğuk koşunun olayları:

- Toplam 116 olay: 46 MCP araması, 1 REST araması, 69 çağrı.
- Sonuçlar: 65 `ok`, 2 `tool_error`, 1 `refused`, 1 `unknown_tool`.
- Bağlar: 2 `search_id`, 64 `recent`, 3 `none`. `none` olanlar: bilinmeyen tool, hiç aranmamış bir
  tool ve stdio'daki echo.
- İstek metni yazılmadı, anahtar dosyası 0600.
- Oturumlar: stdio'da tek bir oturum var (`stdio`). REST'te `X-Session-Id: e2e` → `rest:e2e`.
  HTTP'deki MCP istemcisinin (SDK 2.2, yeni protokol sürümü) oturum kimliği yok → `session: null`.

## Komutlar

```bash
# Mac, repo kökü; embedding'ler GB10'dan (Tailscale)
# $GB10: GB10'un Tailscale adresi (depoda tutulmuyor). Embedding cache'i URL'in yazımıyla anahtarlı: hep aynı adres.
cp -R data/mytools data/w3 && uv run toolrank ingest drop everything-http --out data/w3
uv run toolrank ingest openapi data/specs/github.json --name github --out data/w3   # call_tool için sorgu stilleri
uv run toolrank ingest openapi data/specs/stripe.json --name stripe --out data/w3
uv run toolrank ingest openapi data/specs/petstore-expanded.yaml --out data/w3
# soğuk açılış (DATA/index ve DATA/cache yok) + stdio + REST + günlük
uv run python scripts/serve_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/serve_e2e_w3.json
# cache dolu, indeks yok
mkdir data/w3warm && cp data/w3/tools.jsonl data/w3/sources.json data/w3warm/ && cp -R data/w3/cache data/w3warm/
uv run python scripts/serve_e2e.py --data data/w3warm --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --skip-stdio --out results/serve_e2e_w3warm.json
# ingest'in ısıtması serve'ün cache'iyle aynı mı: "embedded 0 … 1862 already cached"
uv run toolrank ingest openapi data/specs/petstore-expanded.yaml --out data/w3 --emb-url http://$GB10:8091/v1
```

## Ortam

- Mac: Python 3.13; `mcp` 2.2.0, `httpx2` 2.13.1, Starlette 1.7, uvicorn 0.54.
- GB10: vLLM pooling (NGC imajı), Qwen3-Embedding-8B, `--max-model-len 8192`, bf16, port 8091,
  Tailscale üzerinden (`$GB10`).
- Head'ler: `toolrank-heads-qwen3-emb-8b-v0.1.npz` (fp16, sha256 `f3c10125…72f0`).
- Commit: soğuk koşu `4c85887`, cache-dolu koşu `372bba9`.

## Sapmalar ve açıklamalar

- **Düzeltilen: servis zaman aşımı.** Serve embedding isteklerine 10 sn ve tek deneme veriyordu.
  Bu, ilk indeks kurulumunun 32'lik uzun metin batch'lerine de uygulanıyordu. Büyük katalogda ilk
  açılış indekssiz kalabilirdi. Kısıt artık yalnız sorgu gömmelerinde (10 sn, bir tekrar);
  indeksleme 600 sn'yi koruyor. Son başarısız denemeden sonra da artık beklenmiyor (`4c85887`).
- **Düzeltilen: cache yeri.** `ingest --emb-url` cache'i `.cache/toolrank`'a, çağıranın kırpma
  değeriyle yazıyordu. Serve ise `DATA/cache`'i 8192 kırpmayla okuyordu. Isıtma serve'e hiç yaramıyordu.
  Artık üç komut da `DATA/cache`'i paylaşıyor. Isıtma varsayılan olarak serve'ün modelini ve
  kırpmasını kullanıyor (`372bba9`).
- **Düzeltilen: HTTP oturumları.** Oturum kimliği olmayan HTTP istemcileri tek bir `"http"`
  oturumunda toplanıyordu. Artık `session: null`. `recent` bağı da yalnız aynı arayüzün
  aramalarına kuruluyor, böylece REST aramaları MCP çağrılarına bağlanmıyor.
- **Sınır: çağrı–arama bağı.** Yeni protokol sürümündeki HTTP istemcilerinin oturumu yok. Bağ ya
  ajanın geri verdiği `search_id`'ye (tool açıklaması bunu istiyor) ya da `recent`'e kalıyor. E2E'de
  64/69 çağrı `recent` ile bağlandı. Aynı sunucuda eşzamanlı birçok istemci varsa `recent` yanlış
  aramaya bağlayabilir. Faz 2 bağları yöntemine göre ağırlıklandırmalı; istemci kimliği API
  anahtarından (tenant) gelebilir. *Aynı gün kapatıldı, bkz. Ek.*
- **Sınır: soğuk açılış.** Cache boşsa arama 1.862 tool'da ~2.5 dk hata veriyor. Önerilen yol
  `ingest --emb-url`. Dense indeks kurulurken BM25 ile cevap vermek backlog'a yazıldı. *Aynı gün
  kapatıldı, bkz. Ek.*
- **Sınır: OpenAPI yürütücüsü.** Multipart, cookie ve header parametreleri, ham gövde henüz yok
  ("not supported yet"). Bu haftadan önce ingest edilmiş satırlar dizi/nesne sorgu değerlerinde
  "re-ingest" diyor.
- **Bilerek yapılmayan: gerçek GitHub/Stripe çağrısı.** GET'ler kimlik bilgisi olmadan dışarı
  giderdi; POST'un reddi denendi.

## Ek: iki sınır kapatıldı (30 Eyl 2026)

Yukarıdaki "Sınır" maddelerinden ikisi aynı gün kapatıldı. Soğuk açılış yeni kodla yeniden ölçüldü:
`data/w3`'ün cache'siz ve indekssiz bir kopyası, `results/serve_e2e_w3cold_v2.json`.

### İlk açılışta anahtar kelime yedeği

- Anlamsal indeksle paralel bir BM25 indeksi kuruluyor: 1.862 tool'da 0.2 sn. Tool metni
  indekslendiği gibi, istek talimatsız giriyor.
- Anlamsal indeks hazır olana dek:
  - `search_tools` anahtar kelime eşleşmeleri döndürüyor. MCP sonucunda bunu söyleyen bir not,
    REST'te `mode: lexical` var.
  - `call_tool` ve `/v1/tools` çalışıyor.
  - `/v1/rank` beklemeden 503 dönüyor; `/healthz` 503 ve `mode: lexical`.
- İlk kurulum başarısız olursa (ör. embedding sunucusu kapalı) istek geldikçe 30 sn'de bir yeniden
  deneniyor; bu arada yedek cevap veriyor. Değişen bir kataloğun başarısız yeniden kurulumu da
  artık her istekte değil, 30 sn'de bir deneniyor.
- `tools.jsonl` 50 ms değişmeden kalınca tek seferde okunuyor. Böylece yerinde yazılan bir dosya
  yarım indekslenmiyor; testte bu yarış görüldü ve boş bir ara durum yayınlanıyordu.
- BM25 sıralaması thread'ler arasında kilitli, çünkü PyStemmer nesnesi paylaşılamıyor.
  `--hybrid`'in BM25 kolu da bundan yararlanıyor.

| Soğuk açılış (1.862 tool) | Önce | Şimdi |
| --- | --- | --- |
| MCP `list_tools` | 0.88 sn; açıklama "the tools" | 1.15 sn; "the 1862 tools of 5 servers and APIs" |
| İlk `search_tools` | 30 sn bekledi, hata | 1.15 sn'de anahtar kelime sonuçları (6 ms) |
| İndeks kurulurken `call_tool` | 30 sn bekledi, hata | çalışıyor (petstore, 11 ms) |
| Anlamsal indeks hazır | 154 sn | 146 sn |

Anahtar kelime sonuçları anlamsal sonuçlardan zayıf, ama boş dönmüyor (ilk 3):

| İstek | Anahtar kelime (BM25) | Anlamsal |
| --- | --- | --- |
| what time is it in Tokyo right now | `time/convert_time`, 2 alakasız Stripe işlemi | `time/get_current_time`, `time/convert_time` |
| echo back this message | `everything/echo`, `get-annotated-message`, `github/repos/merge` | `everything/echo` |
| list the pets tagged dog | `addPet`, `find pet by id`, `deletePet` (`findPets` 5.) | `findPets`, `addPet` |
| open an issue in my repository … | `search/issues-and-pull-requests`, `issues/add-assignees`, `issues/list-for-repo` | `issues/create` ilk sırada |
| refund the last payment … | `PostPaymentRecordsIdReportRefund`, `PostRefundsRefundCancel`, `PostRefunds` | benzer; `PostRefunds` 2. |
| cancel the subscription … | iki abonelik iptali (`DELETE`), `PostBillingMeterEventAdjustments` | aynı iki iptal, `PostSubscriptionsSubscriptionPause` |

Anlamsal aramada sorgunun gömülmesi bu koşuda 140–554 ms sürdü (ağ ve GB10 yükü; ilk koşuda
131–194 ms). Önbellekteki sorguyla süre değişmedi: p50 5.5 ms.

### Çağrı–arama bağı ve istemci kimliği (günlük şeması v2)

- Her olayda artık bir `client` alanı var: API anahtarının adı | istemci uygulaması (MCP
  `clientInfo`) | uzak adres. Günlüğe HMAC özeti olarak yazılıyor, adres açık yazılmıyor.
- `--api-keys keys.json` ile her istemciye ya da takıma ayrı bir anahtar veriliyor
  (`{ad: anahtar}`, `${ENV}` açılıyor). Anahtarın adı olayın `tenant` alanına giriyor.
- Bağ sırası:
  1. `search_id`;
  2. oturumun aramaları (artık yalnız son aramaya değil, oturumun hepsine bakılıyor);
  3. aynı istemcinin aynı arayüzden aramaları (`client`);
  4. `none`.
- v1'in `recent` bağı (herhangi bir istemcinin son araması) kaldırıldı.

| Bağ (E2E) | v1 (69 çağrı) | v2 (70 çağrı) |
| --- | ---: | ---: |
| `search_id` | 2 | 2 |
| `client` | — | 66 |
| `recent` | 64 | — |
| `none` | 3 | 2 |

- v2'de `none` kalan iki çağrı: bilinmeyen tool (`nope/missing`) ve stdio'daki echo. O süreçte
  echo'yu döndüren bir arama yok.
- HTTP olaylarının hepsi (120) `tenant: e2e` taşıyor; stdio olaylarında (4) tenant yok.
- Günlükte üç ayrı `client` var: MCP istemcisi, REST istemcisi ve stdio.

```bash
mkdir data/w3cold && cp data/w3/tools.jsonl data/w3/sources.json data/w3cold/
uv run python scripts/serve_e2e.py --data data/w3cold --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/serve_e2e_w3cold_v2.json
```

## Sonraki hafta

Faz 1 Hafta 4, platform kancaları. Anthropic tarafında `tool_reference` döndüren özel arama tool'u,
OpenAI tarafında `execution: "client"` modunda `tool_search`. İkisi de bu haftanın `/v1/search`'ü
üzerine kurulacak; örnekler `examples/` altına.
