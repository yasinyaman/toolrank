# Faz 2 — Hafta 6 raporu (2 Ekim 2026): metrikler, kiracılar, Helm

## Sonuç (tek cümle)

Hafta 6'nın üç kutusu kapandı: `GET /v1/metrics` (gecikme, token tasarrufu tahmini, önbellek
isabeti), `--api-keys` üzerinden kiracılar (anahtar başına kaynak izin listesi ve kaynak başına kendi
kimlik bilgileri) ve bir Helm chart'ı; chart kind kümesinde gerçek kurulumla 10/10 denetimden geçti,
GPU modları yalnız render ve API sunucusu doğrulamasıyla denendi.

## Ölçüler, birimler ve yön

- **Token tasarrufu tahmini ↑ %**: `saved / (saved + returned)`; bir tool = adı + açıklaması + giriş
  şeması JSON olarak, 4 karakter = 1 token. Tahmin ve alt sınır (sonraki sonuçların kısaltılmış
  şemaları tam sayılıyor).
- **Smoke denetimi**: `scripts/helm_smoke.py`'nin geçti / toplam sayısı.

## Tablo 1 — metrikler, gerçek sunucuda (`data/w3`, 1.862 tool, 5 kaynak, Mac; embedding GB10'da)

| Ölçü | Değer |
| --- | ---: |
| Kataloğun tahmini boyutu | 909.753 token |
| 4 aramada döndürülen | 21.660 token (32 tool, arama başına ~5,4 bin) |
| 4 aramada tasarruf | 2.715.455 token |
| Tasarruf oranı ↑ | %99,2 |
| Arama süresi (sorgu vektörü önbellekte) | 2,7–14 ms |

## Tablo 2 — Helm chart, kind üzerinde (Mac, Docker Desktop 4 CPU / 4 GB, Kubernetes 1.35)

| İmaj | Denetim | Sonuç |
| --- | --- | ---: |
| `ghcr.io/yasinyaman/toolrank:0.1.0` (yayımlanmış) | kurulum (ingest init + indeks, 16 sn'de hazır), arama, pod içindeki stdio MCP sunucusuna çağrı, yanlış token 401, upgrade ile yeni sunucu katalogda, uninstall veri hacmini tutuyor | 6/6 (metrik ucu o sürümde yok) |
| bu depodan kurulan `toolrank:dev` (`0.2.0.dev0`) | yukarıdakiler + metrikler + tek kaynağa sınırlı anahtar yalnız o kaynağı görüyor, başka kaynağın tool'unu çağıramıyor (404), metrikleri okuyamıyor (403) | 10/10 |
| üç mod (`vllm`, `bundled`, `external` + kiracılar) | `helm template … \| kubectl apply --dry-run=server` | hepsi geçerli |

## Kiracılar: ne yapıldı

- `--api-keys` girdisi artık ya düz anahtar ya da `{key, sources, headers, env}`. `sources` aramayı,
  `/v1/tools`'u, id ile `/v1/rank`'i ve çağrıları o kaynaklarla sınırlıyor; dışarıdaki bir tool
  olmayan bir tool gibi yanıtlanıyor (adı doğrulanmıyor) ve backend'e hiç gitmiyor.
- `headers` / `env` o anahtarın o kaynağa çağrılarına config'inkilerin üstüne ekleniyor; böyle bir
  kaynak anahtar için kendi bağlantısını (stdio'da kendi sürecini) alıyor. Testte: kiracının `env`'i
  yalnız kendi MCP sürecinde görünüyor, paylaşılan süreçte görünmüyor (iki ayrı pid).
- Birlikte-kullanım tabloları ve token tahmini kiracı başına; `/v1/metrics` sınırlı anahtarlara kapalı.

## Komutlar

```bash
# Tablo 1 (Mac; $GB10: GB10'un Tailscale adresi, depoda tutulmuyor)
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz uv run toolrank serve --data data/w3 --port 18792 \
  --api-key "$KEY" --emb-url http://$GB10:8091/v1
curl -s -H "Authorization: Bearer $KEY" -d '{"query": "refund this payment"}' http://127.0.0.1:18792/v1/search
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:18792/v1/metrics

# Tablo 2 (Mac)
kind create cluster --name toolrank
uv run python scripts/helm_smoke.py
docker build -f deploy/docker/Dockerfile -t toolrank:dev . && kind load docker-image toolrank:dev --name toolrank
uv run python scripts/helm_smoke.py --image toolrank:dev --tenants
for m in vllm bundled; do helm template t deploy/helm/toolrank --set auth.apiKey=x --set embedding.mode=$m \
  --set embedding.bundled.image=reg/toolrank-vllm:1 | kubectl apply --dry-run=server -f -; done
kind delete cluster --name toolrank
```

## Ortam

Mac (Docker Desktop 29.8, kind 0.31, Helm 4.2, kubectl 1.36); kod `ea02adb` … `c94f1f7`.

## Sapmalar ve açıklamalar

- **k3s GB10'da denenmedi.** Seçilen yol buydu ama k3s bir sistem servisi ve kurulumu `sudo`
  parolası istiyor; ayrıca GB10'un belleği o sırada 121 GB'ın 115'inde doluydu (başka koşular). Yerine
  Mac'te kind (Docker içinde Kubernetes) kullanıldı: aynı chart, gerçek bir API sunucusu ve kubelet.
  GPU'lu modları (`vllm`, `bundled`) gerçekten çalıştırmak için GB10'da k3s gerekiyor; komutlar
  kullanıcıya verildi.
- **Ortaya çıkan bir sızıntı kapatıldı:** OpenAPI çağrılarının ortak HTTP istemcisi bir çerez
  kavanozu tutuyordu; bir API yanıtının `Set-Cookie`'si o host'a sonraki bütün çağrılara (başka
  istemcilerinkine de) gidiyordu. Artık hiçbir çerez tutulmuyor.
- **Chart tuzağı:** Helm değerleri birleştirdiği için örnek `time` sunucusu, kullanıcı kendi
  `config`'ini verse de kalıyor; `mcpServers: {time: null}` ile düşürülüyor (values ve rehberde yazılı).
- Metrik sayaçları süreçle sıfırlanıyor, sunucu geneli; etiketlerde istek metni, argüman ya da anahtar
  adı yok.

## Sonraki hafta

- Hafta 7–8: pilotlar (design partner'lar) — kod dışı; teknik tarafta pilot ortamında chart'ın GPU modu.
- Kullanıcıda: GB10'da k3s ile `vllm` modunun gerçek denemesi; gerçek trafikle ilk `learn` + `ab` turu.
