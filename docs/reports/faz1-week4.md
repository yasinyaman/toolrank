# Faz 1 — Hafta 4 raporu (30 Eyl 2026)

Takvimde 23–27 Kasım; erken yapıldı.

## Sonuç (tek cümle)

toolrank artık iki büyük ajan API'sinin tool arama kancasının arkasında çalışıyor: Claude'un
Messages API'si (`defer_loading` + `tool_reference`) ve OpenAI'ın Responses API'si (`tool_search`,
istemci tarafında). İkisi de canlı denendi:

- Claude'da üç görevin üçü de doğru çıktı; API'nin kendi BM25 aramasına göre %37–60 daha az girdi
  token'ı harcandı.
- `gpt-5.5`'te iki görev 3 turda doğru çıktı. Üçüncüsü, hesabın dakika başına 10 bin token (TPM)
  sınırına takıldı.

## Ne yapıldı

- **API adları (`names.py`).** Tool id'leri (`github/issues/create`, `swagger-petstore/find pet by
  id`) iki API'nin de kabul ettiği adlara çevriliyor (`^[A-Za-z0-9_-]{1,64}$`).
  - Okunabilir id'ler biçimini koruyor: `/` yerine `__` geliyor (`github__issues__create`).
  - Diğerleri kısaltılıp `___` ve sha256'nın 12 hex karakteri ekleniyor.
  - Okunabilir adlarda üç `_` hiç yan yana gelmiyor, yani eşleme kurgusu gereği bire bir.
  - Ad yalnız id'ye bağlı; katalog değişince bir tool'un adı değişmiyor.
  - `data/w3`'te 1.753 okunabilir, 109 hash'li ad çıktı; çakışma yok.
- **Güvenlik.** `/v1`'e Origin kontrolü eklendi (403). Token'sız bir loopback sunucusuna bir web
  sayfası `text/plain` POST gönderebiliyordu ve Host kontrolü bunu durdurmuyordu; SDK Origin'i
  yalnız `/mcp`'de denetliyor. `/v1/call` yalnız JSON kabul ediyor (415). OpenAPI yol
  parametrelerinde `.` ve `..` reddediliyor: httpx2 bunları kimlik bilgisi giden host'ta üst yola
  çözüyordu.
- **REST.**
  - `POST /v1/call` tool'u MCP `call_tool` ile aynı yoldan çalıştırıyor (`dispatch_call` ortak):
    yazma politikası, şema ipucu ve kullanım günlüğü aynı. Başarısız tool 200 + `isError` dönüyor.
  - Her tool'un `api_name`'i ve `kind`'ı REST'te geliyor; OpenAPI tool'larında HTTP metodu da var.
  - `GET /v1/tools?full=true`, platform istemcisinin ihtiyaç duyduğu kataloğu tek seferde veriyor:
    açıklama, `$schema`'sız ve her zaman nesne olan giriş şeması, notlar ve katalog hash'i.
  - REST oturumları anahtar adının altında tutuluyor.
- **İstemci (`client.py`).** Standart kütüphaneyle yazılmış bir REST istemcisi (`health`, `catalog`,
  `search`, `call`):
  - Yönlendirme izlemiyor, böylece token başka host'a gitmiyor.
  - Çağrıyı asla tekrar denemiyor.
  - Hatalar durum koduyla `ToolrankError` olarak geliyor.
- **Entegrasyonlar (`integrations/`).** İkisi de vendor SDK'larını import etmiyor: SDK nesnelerini
  de dict'leri de okuyor, geri düz dict gönderiyor.
  - **Anthropic** (`Toolbox`, `run`):
    - Bütün katalog `defer_loading` ile gidiyor; görünen tek tool `search_tools`, toolrank onu
      yalnız `tool_reference` bloklarıyla cevaplıyor.
    - Referanslar yalnız gönderilen kataloğun adlarını içeriyor; bilinmeyen bir ad bütün isteği
      400 ile düşürür.
    - Tool listesi konuşma başında donduruluyor, geçmişe yalnız ekleme yapılıyor.
    - Her `tool_use` tek mesajda, sırayla cevaplanıyor. `max_tokens` ve `refusal` sonrasında hiçbir
      şey çalışmıyor; `pause_turn` olduğu gibi geri gönderiliyor.
    - `builtin="bm25"` API'nin kendi aramasını kullanıyor.
  - **OpenAI**:
    - İstekte yalnız `tool_search` (`execution: "client"`) var.
    - Her `tool_search_call`, bulunan tool'ların tam `function` tanımlarıyla (`defer_loading`,
      `strict: false`) cevaplanıyor; her tool konuşma başına bir kez gönderiliyor.
    - Döngü durumsuz: `store=False`, şifreli reasoning, bütün öğeler her turda yeniden gidiyor.
    - `failed` hata fırlatıyor; `incomplete` bir cevaptaki hiçbir çağrı çalıştırılmıyor.
    - Bozuk JSON argüman, yüklenmemiş ad ve reddedilen çağrı, modelin okuyacağı bir çıktıya
      dönüşüyor.
- **Örnekler.** `examples/anthropic_tool_reference.py` ve `examples/openai_client_tool_search.py`:
  - Anahtarlar ortamdan ya da `.env`'den geliyor, hiç yazdırılmıyor.
  - Yazma yapabilecek çağrılar `--yes` verilmedikçe soruluyor (`readOnlyHint` olmayan tool ya da
    GET/HEAD dışı işlem): bir tool çıktısı modeli yönlendirebilir.
- **Testler.** Gerçek `anthropic` 1.9.0 ve `openai` 3.22.1 SDK'ları `httpx2.MockTransport`
  üzerinden, toolrank da süreç içinde (stdio fixture MCP sunucusu arkada) koşuyor. Kontrol edilenler:
  - SDK'nın gerçekte gönderdiği JSON;
  - bizim dict'lerin SDK'nın kendi tiplerine uyması (`pydantic.TypeAdapter`);
  - değişmezler: geçmiş yalnız büyüyor, her turda aynı tool listesi, her çağrıya tek cevap.

  Toplam 119 test geçiyor, 7'si atlanıyor.

## Çevrimdışı kontrol (`data/w3`, 1.862 tool)

| Ölçüm | Değer |
| --- | --- |
| Katalog indirme (`?full=true`) | 47 ms |
| Anthropic isteği başına ertelenmiş tanımlar | 3,67 MB (sınır 32 MB; önbellek önekinin dışında) |
| JSON Schema 2020-12 meta-şemasına uymayan giriş şeması | 0 |
| OpenAI'a yüklenen tanımlar: Tokyo / köpekler / iade araması | 1,2 KB / 2,2 KB / 20,7 KB |
| `/v1/call`: petstore GET / Stripe POST | `ok` / `refused` |

Canlı koşularda bütün OpenAPI kaynakları (petstore, Stripe, GitHub) yerel bir taklide yönlendirildi,
dolayısıyla hiçbir tool çağrısı makineden çıkmadı. Taklit petstore'un `/v2/pets` yolunu cevaplıyor,
diğer her yol 404. Stripe yazmaları proxy'de reddediliyor (`--allow-write` yok).

## Canlı: Claude (`claude-opus-5-5`)

Her görev iki kurulumla koşuldu. İkisinde de ertelenmiş tool'lar ve çağrı yolu aynıydı; yalnız
arama farklıydı.

| Görev | Arama | Tur | Süre | Girdi / çıktı token | Aramalar → çağrılar | Cevap |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Tokyo'da saat kaç? | toolrank | 3 | 16,2 sn | 3.592 / 211 | 1 arama (3 tool) → `time/get_current_time` | doğru (13:51 JST) |
| | API BM25 | 2 | 9,6 sn | 5.833 / 197 | 1 arama (5 tool; 3'ü Stripe) → aynı çağrı | doğru |
| Köpek etiketli hayvanlar | toolrank | 3 | 12,6 sn | 4.216 / 186 | 1 arama (3 petstore tool'u) → `findPets` | doğru (Rex, Ace) |
| | API BM25 | 2 | 9,2 sn | 6.645 / 186 | 1 arama (4 petstore + `github/git/create-tag`) → `findPets` | doğru |
| cus_123'ün son ödemesini iade et | toolrank | 7 | 38,2 sn | 92.747 / 1.163 | 4 arama → 5 GET (hepsi taklitten 404) | "ödeme bulunamadı, hiçbir şey iade edilmedi" |
| | API BM25 | 4 | 34,7 sn | 231.134 / 1.178 | 5 arama → 3 GET (404) | aynı sonuç |

- **Token.** toolrank'ın aramasıyla girdi token'ı %37 (Tokyo), %37 (köpekler) ve %60 (iade) daha
  az. Fark yüklenen tanımlardan geliyor: BM25 alakasız Stripe işlemleri de yüklüyor (Stripe
  şemaları 56 KB'a kadar çıkıyor). toolrank'ın uyarlanabilir K'sı net isteklerde 3 tool döndürüyor.
- **Tur ve süre.** İstemci tarafı aramanın bedeli, her arama için bir API turu daha. API'nin kendi
  araması aynı istek içinde sunucuda çalışıyor. Basit görevlerde bu, 3'e karşı 2 tur ve 16,2'ye
  karşı 9,6 sn demek. İade görevinde toolrank daha çok tur attı (7'ye 4) ama süre yakın kaldı
  (38,2'ye 34,7 sn).
- **İade görevi.** Bu kurulumda başarılı olamazdı (Stripe GET'leri 404, POST'lar reddediliyor);
  modelin davranışını ölçüyor. İki kurulumda da model ödeme kimliğini bulamayınca körlemesine iade
  denemedi ve bunu söyledi.
- **İlk çağrı.** Zamanlayıcı sunucusuna ilk çağrı `uvx` sürecinin başlamasını içeriyor (1,2 sn).

## Canlı: OpenAI (`gpt-5.5`)

İlk denemede üç görev de 429 `insufficient_quota` ile döndü: hesapta kredi yoktu. Kredi
eklendikten sonra yalnız OpenAI kısmı yeniden koşuldu (`--setups gpt+toolrank`).

| Görev | Tur | Süre | Girdi / çıktı token | Aramalar → çağrılar | Cevap |
| --- | ---: | ---: | ---: | --- | --- |
| Tokyo'da saat kaç? | 3 | 6,4 sn | 1.004 / 112 | 1 arama, 1 tool yüklendi → `time/get_current_time` | doğru (14:01 JST) |
| Köpek etiketli hayvanlar | 3 | 4,7 sn | 1.775 / 106 | 1 arama, 2 tool yüklendi → `findPets` | doğru (Rex, Ace) |
| cus_123'ün son ödemesini iade et | — | 59,1 sn | — | 2 arama; 10 + 8 Stripe tool'u yüklendi | 429: TPM sınırı |

- **İlk iki görev.** Claude'dan kısa sürdü (6,4'e karşı 16,2 sn ve 4,7'ye karşı 12,6 sn). Az token
  harcandı, çünkü yüklenen tanım az: uyarlanabilir K, Tokyo isteğinde tek tool döndürdü.
- **İade görevi.** Model iki geniş arama yaptı ve 18 Stripe işlemini tam şemalarıyla yükledi; bazı
  şemalar 50 KB civarında. Üçüncü istek 34.551 token'a çıktı. Hesabın sınırı dakikada 10.000 token
  (ilk kademe), bu yüzden istek reddedildi. SDK iki kez yeniden denediği için hata 59 sn sonra geldi.
- **Sebep.** Bu bir hesap kademesi sınırı, ama gösterdiği şey kalıcı: geniş aramalarda büyük
  şemalar bağlamda birikiyor. OpenAI tarafında yüklenen tanımlar sonraki her turda yeniden
  gönderiliyor.
- **Önlem.** İlk birkaç sonuç dışındaki tool'ları MCP proxy'deki gibi kısaltılmış şemayla yüklemek
  ya da platformlarda K'nın üst sınırını düşürmek. Backlog'a yazıldı.

## Kullanım günlüğü

Canlı koşunun 14 çağrısı günlüğe düştü:

| Bağ | Sayı | Açıklama |
| --- | ---: | --- |
| `search_id` | 7 | toolrank aramalı koşular `search_id`'yi geri veriyor |
| `client` | 7 | BM25 koşularının çağrıları |

`client` ile bağlanan 7 çağrıda aramayı toolrank yapmamıştı. Bu çağrılar aynı istemci anahtarı
üzerinden başka konuşmaların toolrank aramalarına bağlandı; bu yanlıştı ve düzeltildi. Artık oturumu
olan bir çağrı başka bir oturumun aramasına hiç bağlanmıyor; `client` bağı yalnız oturumsuz çağrılar
için. Düzeltmeden sonra bu 7 çağrı `none` olurdu.

Düzeltmeden sonraki OpenAI koşusunda 4 çağrı var:

| Bağ | Sayı | Açıklama |
| --- | ---: | --- |
| `search_id` | 2 | OpenAI koşularının çağrıları |
| `client` | 2 | Betiğin çevrimdışı bölümündeki oturumsuz iki çağrı, aynı istemcinin az önceki aramalarına bağlandı (doğru) |

## Komutlar

```bash
# Mac, repo kökü; toolrank serve'ü betik kendisi başlatıyor (data/w3, GB10 embedding'leri)
# $GB10: GB10'un Tailscale adresi (depoda tutulmuyor). Embedding cache'i URL'in yazımıyla anahtarlı: hep aynı adres.
uv run python scripts/platforms_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/platforms_e2e_offline.json
# canlı (anahtarlar .env'den): üç görev x Claude+toolrank, Claude+BM25, OpenAI+toolrank
uv run python scripts/platforms_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --live --out results/platforms_e2e_live.json
# yalnız OpenAI (kredi eklenince)
uv run python scripts/platforms_e2e.py ... --live --setups gpt+toolrank --out results/platforms_e2e_gpt.json
```

## Ortam

- Mac: Python 3.13; `anthropic` 1.9.0, `openai` 3.22.1.
- Modeller: `claude-opus-5-5`, `gpt-5.5` (OpenAI hesabı ilk kademede: 10 bin TPM).
- Embedding'ler: GB10'daki Qwen3-Embedding-8B, Tailscale üzerinden. Head'ler fp16 `.npz`.
- Commit'ler: `863af03` … `fe6d1d0`, ardından bu rapor.

## Sapmalar ve açıklamalar

- **OpenAI iade görevi.** Hesabın 10 bin TPM sınırına takıldı (istek 34,5 bin token). İlk iki görev
  doğru çalıştı; iade görevi daha yüksek bir kademede ya da tanımlar kısaltılarak yeniden koşulmalı.
- **Anthropic isteklerinin boyutu.** Katalog her istekle gidiyor (3,67 MB). Token'a yansımıyor ama
  bant genişliğine yansıyor. Yeni `inline-tools-2026-09-15` beta'sı (`tool_addition` blokları) tool'ları
  konuşmanın içinde, değerle tanımlıyor; kataloğun gönderilmesine gerek kalmıyor. Backlog'da.
- **Fazladan tur.** İstemci tarafı arama her arama için bir tur ekliyor. Aynı beta'yla ilk kullanıcı
  mesajına, toolrank'ın göreve göre bulduğu tool'lar baştan eklenebilir; basit görevlerde tur farkı
  kapanır. Backlog'da.
- **Düzeltilen.** Oturumu olan çağrıların başka konuşmaların aramasına `client` üzerinden bağlanması.

## Sonraki hafta

Faz 1 Hafta 5, framework adapter'ları ve fine-tune CLI:

- langgraph-bigtool `retrieve_tools_function`, LlamaIndex `ObjectRetriever`, LiteLLM MCP filtresi
  (`ToolrankClient` bunların tabanı);
- `toolrank finetune`;
- `toolrank eval` çıktısından README tablosu.
