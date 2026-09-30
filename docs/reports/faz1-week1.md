# Faz 1 — Hafta 1 raporu (29 Eyl 2026)

Takvimde 2–6 Kasım; kapı kararından sonra erken başlandı.

## Sonuç (tek cümle)

`toolrank ingest` MCP sunucularını (stdio ve streamable HTTP) ve OpenAPI 3.x spec'lerini, Faz 0'da
ölçülen metin biçimiyle tek bir dizine indeksliyor. Gerçek kaynaklarla denendi: üç resmi MCP
sunucusu (16 tool, 2.5 sn) ve GitHub REST + Stripe (1.843 işlem, her biri 0.1 sn'nin altında).
Tekrar çalıştırma hiçbir şeyi yeniden gömmüyor (0 token, 0.3 sn); tek bir değişiklik yalnız o
tool'u gömüyor. Stargate spec'leriyle test ertelendi (backlog'da); hafta kapandı.

## Ne yapıldı

- **Metin biçimi (`ingest/text.py`).** Her ingest edilen tool
  `{"server", "name", "description", "inputSchema"}` JSON'u ile indeksleniyor. Bu, Faz 0'da sunucu
  adının MCP-Zero'da top-1'e +8 puan getirdiği ablasyonun birebir biçimi (`documentation` formatı
  bunu döndürüyor). 6.000 karakterlik bir bütçe var. Bütçe aşılınca önce şema küçülüyor (derinlik,
  iç açıklamalar, ilk 8 özellik), sonra açıklama; sunucu ve ad hiç kesilmiyor. JSON-schema'nın
  anlamsız `$schema` alanı metinden atılıyor.
- **MCP (`ingest/mcp.py`, `adapters/mcp_client.py`).** `--server NAME=URL|KOMUT` ya da MCP
  istemcilerinin kendi dosyası (`mcpServers`: Claude Desktop/Cursor; `servers`: VS Code) okunuyor.
  Resmi SDK (`mcp` 2.2, `mcp.Client`) protokol sürümünü eski ve yeni sunucularla müzakere ediyor.
  Bütün sayfalar okunuyor; sunucular eşzamanlı listeleniyor ve her birine başlangıçtan son sayfaya
  tek bir zaman aşımı uygulanıyor. Hata olursa sunucunun stderr'inin sonu gösteriliyor. Env
  değişkenleri ve header'lar hiçbir dosyaya yazılmıyor.
- **OpenAPI (`ingest/openapi.py`).** Bir işlem bir tool'a dönüşüyor.
  - Ad `operationId`'den, açıklama özet + açıklamadan (HTML ayıklanmış) geliyor.
  - `inputSchema` tek düz bir nesne: path, query, header ve cookie parametreleriyle body'nin üst
    düzey alanları.
  - Body'de önce JSON, sonra form (Stripe'ın bütün body'leri form), sonra multipart seçiliyor.
    `allOf` birleşiyor, `oneOf`/`anyOf` birleşim olarak alınıyor, boş nesne body'ler atlanıyor
    (Stripe her GET'e bir tane koyuyor). Çakışan parametre `ad__<konum>` oluyor.
  - Yerel `$ref`'ler döngü korumasıyla açılıyor.
  - Çağırma bilgisi (method, path, base URL, argüman konumları) Hafta 3'ün proxy'si için
    `Tool.doc["http"]`'de, indekslenen metnin dışında duruyor.
  - Swagger 2.0 dönüştürme ipucuyla reddediliyor.
- **Senkron (`ingest/sync.py`).**
  - Dizin `tools.jsonl` + `sources.json` (tür, tool sayısı, son senkron, spec kaynağı; sır yok).
  - Listelenen kaynak yalnız kendi tool'larını değiştiriyor; hata veren kaynak eskilerini koruyor.
  - Bir ad türüne bağlı: `--replace` olmadan MCP sunucusu ile spec birbirinin yerine geçmiyor.
  - 0 tool dönen liste, `--allow-empty` olmadan kaynağı silmiyor.
  - Aynı listeyi tekrar yazmak bayt bayt aynı dosyayı üretiyor.
  - Embedding cache metinle anahtarlandığı için değişmeyen tool zaten yeniden gömülmüyor;
    `--emb-url/--emb-model` yalnız cache'te olmayanları gömüyor.

## Ölçümler

MCP (Mac; paketler önbellekte; üç sunucu tek komutta, eşzamanlı):

| Sunucu | Taşıyıcı | Tool | En uzun metin (karakter) |
| --- | --- | ---: | ---: |
| `uvx mcp-server-time` | stdio | 2 | 716 |
| `uvx mcp-server-fetch` | stdio | 1 | 1.178 |
| `npx -y @modelcontextprotocol/server-everything` | stdio | 13 | 1.018 |
| aynı sunucu, `streamableHttp` modu | streamable HTTP | 13 (stdio ile aynı adlar) | — |

Üç stdio sunucusu toplam 2.5 sn sürdü. Erişilemeyen bir URL, bulunmayan bir komut, takılan ve
çöken sunucular için hata mesajları testlerde doğrulandı.

OpenAPI (sabit commit'ler `data/specs/SOURCES.txt`'de):

| Spec | İşlem | Dönüşüm | Metin p50 / p90 / p99 / en uzun (bütçesiz) | 4K / 6K / 8K bütçede kısalan |
| --- | ---: | ---: | --- | --- |
| Petstore (expanded, YAML) | 4 | < 0.01 sn | 309 / 1.837 / 1.837 / 1.837 | 0 / 0 / 0 |
| GitHub REST (`api.github.com.json`, 13 MB) | 1.231 | 0.04 sn | 1.078 / 2.360 / 6.694 / 25.183 | 52 / 20 / 11 |
| Stripe (`spec3.json`, 8.3 MB) | 612 | 0.06 sn | 1.258 / 5.962 / 33.197 / 55.966 | 81 / 61 / 48 |

Faz 0'ın MCP metinleri bütçenin çok altındaydı: MCP-Zero'da en uzun 1.523 karakter, LiveMCPBench'te
p99 3.855 (tek bir tool 6.000'i aşıyor). 6.000 bütçesi GitHub'ın %1.6'sını, Stripe'ın %10'unu
kısaltıyor.

Embedding ısıtma (GB10, Qwen3-Embedding-8B bf16, port 8091, `--truncate 8192`, `documentation`):

| Adım | Gömülen metin | Token | Süre |
| --- | ---: | ---: | ---: |
| GitHub, ilk ingest | 1.231 | 390.059 | 1 dk 20 sn |
| Stripe, ilk ingest | 612 (1.231'i zaten cache'te) | 249.557 | 51 sn |
| Stripe, tekrar | 0 | 0 | 0.3 sn |
| Stripe, bir işlemin özeti değişti | 1 (`~1`) | 1.327 | 0.6 sn |

Süreler komutun tamamı: dönüşüm, yazma ve uv başlangıcı dahil. Gömme hızı ~5.2K token/sn.

vLLM'in kırpma yönü (`scripts/truncation_side.py`): 19.754 token'lık bir metin 8.192'ye
kırpıldığında ortaya çıkan vektör, metnin ilk 8.192 token'ının vektörüne 0.9915, son 8.192
token'ınınkine 0.5929 kosinüs benzerliğinde. Yani vLLM 0.13 pooling'de metnin başı kalıyor ve
kırpma sunucu/ad/açıklamayı kaybettirmiyor. Bütçe yine de metni sağlayıcıdan bağımsız kılıyor.

## Komutlar

```bash
# Mac, repo kökü (kod: fc22da8 + bu commit)
uv pip install -e ".[dev]"                                   # mcp 2.2.0, pyyaml
mkdir -p data/specs
curl -sfL -o data/specs/petstore-expanded.yaml https://raw.githubusercontent.com/OAI/learn.openapis.org/bbb743ed3b7c5ed76b6e6ba9b302af38f3956c44/examples/v3.0/petstore-expanded.yaml
curl -sfL -o data/specs/github.json https://raw.githubusercontent.com/github/rest-api-description/2f44eacae7f376f1ed158829fbe008cc4eae5815/descriptions/api.github.com/api.github.com.json
curl -sfL -o data/specs/stripe.json https://raw.githubusercontent.com/stripe/openapi/db67eb25e0ca80f12ffffb8f04dc1ba8289cd10f/openapi/spec3.json

S=(--server time="uvx mcp-server-time" --server fetch="uvx mcp-server-fetch" --server everything="npx -y @modelcontextprotocol/server-everything")
uv run toolrank ingest mcp "${S[@]}" --out data/mytools
uv run toolrank ingest openapi data/specs/github.json --name github --out data/mytools
uv run toolrank ingest openapi data/specs/stripe.json --name stripe --out data/mytools
uv run toolrank ingest openapi data/specs/petstore-expanded.yaml --out data/mytools
uv run toolrank ingest mcp "${S[@]}" --out data/mytools                                  # hepsi "="
uv run toolrank ingest openapi data/specs/petstore-expanded.yaml --name time --out data/mytools  # tür koruması, çıkış 1
PORT=3917 npx -y @modelcontextprotocol/server-everything streamableHttp &
uv run toolrank ingest mcp --server everything-http=http://localhost:3917/mcp --out data/mytools
uv run toolrank ingest drop fetch --out data/mytools

# GB10, ~/toolrank (specs scp ile kopyalandı)
E="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192"
~/.local/bin/uv run toolrank ingest openapi data/specs/github.json --name github --out data/mytools $E
~/.local/bin/uv run toolrank ingest openapi data/specs/stripe.json --name stripe --out data/mytools $E   # iki kez
python3 -c "import json; s=json.load(open('data/specs/stripe.json')); s['paths']['/v1/customers']['post']['summary']='Create a new customer'; json.dump(s, open('/tmp/stripe_changed.json','w'))"
~/.local/bin/uv run toolrank ingest openapi /tmp/stripe_changed.json --name stripe --out data/mytools $E
~/.local/bin/uv run python scripts/truncation_side.py --tools data/mytools/tools.jsonl
```

## Ortam

Mac: Python 3.13, `mcp` 2.2.0, Node 24 (npx), uv 0.11. GB10: vLLM 0.13.0 (NGC 26.01),
Qwen3-Embedding-8B bf16 (`--max-model-len 8192`), embedding cache sıcak (ToolRet) ama bu metinler
için soğuk.

## Sapmalar ve açıklamalar

- **Stargate spec'leri** yok. OpenAPI dönüşümü GitHub, Stripe ve Petstore ile test edildi; Stargate
  testi 30 Eyl'de ertelendi ve backlog'a taşındı, plan kutusu açık spec'lerle kapandı.
- **Kalite ölçülmedi.** Bu set için etiketli sorgu yok; metin biçimi Faz 0'da MCP setlerinde
  ölçülen biçim. OpenAPI'nin doğrulama anahtarları (`maxLength: 5000` gibi) metinde kaldı. Onları
  atmanın ya da bütçenin retrieval'a etkisi Hafta 2'de dizin/arama gelince ölçülebilir.
- **Kapsam dışı:** SSE taşıyıcısı (açık hata mesajı veriyor), Swagger 2.0 (dönüştürme ipucu),
  harici `$ref`'ler (stub'lanıp sayılıyor), VS Code'un `${input:…}` değişkenleri (olduğu gibi
  geçiyor).
- **Plan ajanının öngörüsünün tersine** vLLM başı tutuyor (yukarıda). Faz 0'da CLM'in 2.048 token
  kırpması da tool'un başını tutmuştu; Faz 0 sonuçlarını etkileyen bir durum yok.

## Sonraki hafta

Faz 1 Hafta 2: index port'u (numpy, FAISS, pgvector), hibrit skor (RRF; BM25 kolu talimatsız),
uyarlanabilir K ve kazanan head'in paketlenmesi.
