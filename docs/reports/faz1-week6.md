# Faz 1 — Hafta 6 raporu (30 Eyl 2026)

Takvimde 7–11 Aralık; erken yapıldı.

## Sonuç (tek cümle)

Hafta 6'nın üç maddesi bitti. Herkese açık hiçbir şey yayımlanmadı; yayın, verdiğiniz kararla Hafta
7'de, 19:00'dan sonra ve her adım ayrı onayla yapılacak.

- **Paket:** PyPI'a hazır (`0.1.0.dev0`). Temiz bir ortamda hem yalın hem `[mcp]` kurulumu uçtan uca
  çalıştı.
- **İmajlar:** İki Docker imajı ve iki compose örneği GB10'da uçtan uca çalıştı. vLLM'in resmi
  v0.30.0 imajı GB10'da çalışıyor ve Faz 0'ın sayılarını tekrarlıyor.
- **FP8:** Qwen3-Embedding-8B'nin FP8 kopyası, önceden belirlenen eşiklerin hepsini geçti ve
  varsayılan oldu. Tek sorgu gecikmesi 99 ms'den 55 ms'ye, ağırlık belleği 14.1 GiB'den 7.6 GiB'ye
  indi.
- **Docs ve topluluk:** mkdocs docs sitesi (17 sayfa); lisans, bildirim ve topluluk dosyaları; CI
  ve kapalı duran yayın iş akışları.

## 1. Paket (PyPI'a hazır, yayımlanmadı)

- **Sürüm tek yerde.** Sürüm artık yalnız `toolrank/__init__.py`'de duruyor (`0.1.0.dev0`); hatch
  oradan okuyor.
- **Metadata.**
  - PEP 639 biçiminde lisans: `License-Expression: Apache-2.0`; wheel'de LICENSE ve NOTICE var.
  - Python 3.13 classifier'ı eklendi.
  - Adresler `yasinyaman/toolrank`.
- **İçerik.**
  - sdist yalnız `src/`, `tests/`, README, LICENSE, NOTICE ve CHANGELOG'u taşıyor.
  - wheel 155 KB, sdist 198 KB.
  - `twine check --strict` geçiyor.
- **Kurulum hataları.** Aşağıdakiler traceback yerine kurulum satırı yazıyor:
  - `[mcp]`'siz `serve` ve `ingest mcp`;
  - `[faiss]`/`[pgvector]`'süz indeks;
  - torch'suz `.pt` head.

  Olmayan bir dosyayı gösteren `TOOLRANK_HEADS` artık sessizce head'siz çalışmıyor, hata veriyor.
- **`toolrank heads pull`.** Paketlenmiş head'leri indirip sha256'sını kontrol ediyor ve önbelleğe
  koyuyor. Head'ler barındırılana kadar anlaşılır bir hata veriyor.
- **Konteyner ve ters proxy ayarları.** `serve`, `0.0.0.0`'a bağlanınca loopback adlarını da kabul
  ediyor. Portsuz bir Host başlığı `/mcp`'de de geçiyor. Yeni ortam değişkenleri:
  `TOOLRANK_ALLOWED_HOSTS`, `TOOLRANK_EMB_URL`, `TOOLRANK_EMB_MODEL`. Eval bunları okumuyor.
- **Temiz ortam denemesi.**
  - Yalın kurulumda yalnız numpy ve bm25s var; BM25 eval çalıştı, `serve` kurulum satırını verdi.
  - `[mcp]` kurulumunda `data/w3` (1.862 tool) GB10 embedding'leriyle servis edildi. İndeks 0,2
    sn'de hazırlandı, Tokyo sorusunda arama doğru tool'ları buldu, çağrı çalıştı.
- **Kullanıcıya görünen metinler.** CLI yardımı ve docstring'lerdeki Faz/Hafta ifadeleri temizlendi.

## 2. FP8 backbone

vLLM, `--quantization fp8` ile resmi bf16 ağırlıkları yüklerken FP8'e çeviriyor; üçüncü taraf bir
checkpoint gerekmiyor. Sunucu `deploy/spark/compose.yaml`'ın fp8 profilinde, 8094'te,
`qwen3-emb-fp8` adıyla çalıştı. Eşikler koşulardan önce plana yazılmıştı. Karar head satırına göre
veriliyordu, ama dense satır da geçti.

| Satır | Ölçüt | bf16 | FP8 | Fark | Eşik |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-Emb + head'ler | ToolRet NDCG@10 (micro) | 54.03 | 53.94 | −0.09 | ±0.3 |
| | ToolRet cat-macro | 47.13 | 47.27 | +0.14 | ±0.3 |
| | LiveMCPBench Recall@5 | 53.03 | 53.48 | +0.45 | ±1.1 |
| | MCP-Zero top-1 | 79.87 | 79.51 | −0.36 | ±0.5 |
| Qwen3-Emb | ToolRet NDCG@10 (micro) | 51.11 | 51.05 | −0.06 | ±0.3 |
| | ToolRet cat-macro | 46.54 | 46.42 | −0.12 | ±0.3 |
| | LiveMCPBench Recall@5 | 50.82 | 51.52 | +0.70 | ±1.1 |
| | MCP-Zero top-1 | 78.19 | 78.04 | −0.15 | ±0.5 |

- **Vektör uyumu** (`scripts/fp8_agreement.py`, ToolRet):
  - backbone kosinüsü ortalama 0.998 (en kötü %1'lik dilim 0.997 / 0.996);
  - head'lerden sonra 0.997 (0.993 / 0.992);
  - 1.000 sorguda ilk 10'un %94.5'i, ilk sıradaki tool'un %89.7'si aynı.

  Faz 0'daki CLM FP8'inde bu oranlar %80 ve %74'tü.
- **Gecikme** (`scripts/latency.py`, 44.453 tool'luk ToolRet indeksi, head'lerle, batch 1, 100
  sorgu):
  - bf16: soğuk p50 98.5 ms (p95 123.6), sıcak 7.7 ms;
  - FP8: soğuk p50 54.9 ms (p95 67.6), sıcak 8.1 ms.
- **Bellek:** ağırlıklar bf16'da 14.1 GiB, FP8'de 7.6 GiB.
- **Gömme süresi:** ToolRet korpusu FP8'de 2.033 sn'de gömüldü. GB10'da aynı anda başka işler de
  vardı; bf16 tek başınayken 1.505 sn sürmüştü.
- **Karar:** FP8 imajların ve compose'un varsayılanı oldu. Sonuç tablosuna "FP8 + head'ler (Docker
  varsayılanı)" satırı, model kartına FP8 notu eklendi.
- **Adlandırma:** FP8 her yerde `qwen3-emb-fp8`, bf16 `qwen3-emb` adıyla sunuluyor. Embedding
  önbelleği dtype'a değil ada göre anahtarlandığı için aynı adla iki tür vektör karışırdı.

## 3. vLLM v0.30.0 paritesi

Docker örnekleri NGC imajı yerine vLLM'in resmi `vllm/vllm-openai:v0.30.0` imajını kullanıyor.
NVIDIA'nın NGC imajı yeniden dağıtılamıyor.

- **GB10'da çalışıyor mu:** Resmi imaj (CUDA 13.0, arm64) GB10'da çalıştı: FlashInfer `121a`
  mimarisini tanıdı ve `torch.compile` 31 sn sürdü.
- **Vektörler:** Aynı metinlerde 0.13 ile 0.30'un vektörleri arasındaki kosinüs 0.99993.
- **Sayılar:** MCP setlerinde fark bir sorgunun ağırlığından küçük:

| Set | Satır | 0.13 (NGC) | 0.30 (resmi) | Fark |
| --- | --- | ---: | ---: | ---: |
| LiveMCPBench Recall@5 | Qwen3-Emb | 50.82 | 50.82 | 0.00 |
| | + head'ler | 53.03 | 52.77 | −0.26 |
| MCP-Zero top-1 | Qwen3-Emb | 78.19 | 78.15 | −0.04 |
| | + head'ler | 79.87 | 79.76 | −0.11 |

İstemcimiz v0.30'da kaldırılan `normalize` alanını göndermiyor; `--runner pooling` ve
`truncate_prompt_tokens` duruyor.

## 4. Docker imajları ve compose

- **`toolrank` imajı** (`deploy/docker/Dockerfile`):
  - `python:3.12-slim` üzerinde `toolrank[mcp,openapi,stem]`; sürümler `uv.lock`'tan geliyor, yani
    THIRD_PARTY'dekilerle aynı.
  - stdio MCP sunucuları için Node.js 24 ve uv; alt süreçleri toplamak için tini.
  - Root olmayan kullanıcı (uid 1000); bytecode önceden derlenmiş.
  - Head'ler isteğe bağlı bir `heads` build context'inden geliyor. Bu bağlamdan yalnız
    `toolrank-heads-*.npz` dosyaları alınıyor; GB10'daki Hafta 5 deneme head'i imaja girmedi.
  - arm64'te 505 MB.
- **`toolrank-vllm` imajı** (`deploy/docker/Dockerfile.vllm`):
  - Resmi vLLM imajı üzerinde, toolrank kendi venv'inde, vLLM'in bağımlılıklarıyla çakışmadan.
  - Giriş betiği vLLM'i loopback'te başlatıyor ve bekliyor, sonra toolrank'ı çalıştırıyor. Biri
    düşerse konteyner çıkıyor.
  - FP8 varsayılan (`TOOLRANK_FP8=0` ile bf16).
  - 22.6 GB; bunun 22.1 GB'ı vLLM tabanı.
- **Compose:** `compose.yaml` resmi vLLM ile ince imajı iki servis olarak, `compose.bundle.yaml`
  tek imajı çalıştırıyor. Proje adları `toolrank-stack` ve `toolrank-bundled`: GB10'daki
  `deploy/spark` compose'u `toolrank` adını kullanıyor ve `down --remove-orphans` benchmark
  sunucularını silebilirdi. Veri, model ve uvx/npx önbelleği adlandırılmış volume'lerde; bind
  mount'ta dizini Docker root sahipliğiyle oluşturur ve uid 1000 yazamazdı.
- **Duman testi** (`scripts/container_smoke.py`, GPU'suz, sahte embedding sunucusuyla): ince imaj
  sekiz kontrolün sekizini geçti.
  - sürüm;
  - ortam değişkenleriyle ingest ve ısıtma;
  - sağlık;
  - yayımlanan porttan `localhost` ile arama;
  - yabancı Host'a 421;
  - MCP `initialize` 200;
  - `node`, `npx` ve `uvx` mevcut;
  - head'lerle semantik mod.
- **GB10'da uçtan uca** (ayrı compose projeleri, iş bitince her şey silindi):

| Yığın | Ingest (vLLM açılışı + uvx dahil) | `up` → sağlıklı | Arama | Çağrı |
| --- | ---: | ---: | --- | --- |
| `compose.yaml` | 185 sn | 6 sn | `time/get_current_time`, `time/convert_time` | Tokyo saati |
| `compose.bundle.yaml` | 182 sn | 187 sn (vLLM konteynerde yeniden açıldı) | aynı | Tokyo saati |

Bu betikteki MCP `initialize` isteği 400 aldı. Sebep betiğin kendisi: curl isteğine `Content-Type`
eklenmemişti. Duman testi aynı kodla, doğru başlıkla 200 alıyor.

## 5. Docs sitesi

- **Kurulum:** mkdocs 1.6 ve Material 9.7, `https://yasinyaman.github.io/toolrank/` için. Araçlar bir
  PEP 735 grubunda: `uv run --group docs mkdocs serve`.
- **Sayfalar:**
  - genel bakış;
  - hızlı başlangıç (Docker ve pip, Claude Code / Claude Desktop / REST bağlantıları);
  - kavramlar;
  - altı rehber: ingest, serve, Claude ve OpenAI, framework'ler, Docker, fine-tune;
  - benchmark'lar;
  - mimari (üç mermaid diyagramı);
  - referans: CLI, REST, Python API; model kartı;
  - katkı ve değişiklik günlüğü.
- **Referans:** CLI referansı `scripts/cli_reference.py` ile parser'dan üretiliyor. argparse'ın
  `--help` metni Python sürümleri arasında değiştiği için tablolar parser'ın kendi tanımlarından
  kuruluyor. Python API sayfası mkdocstrings'le docstring'lerden geliyor.
- **Sonuç tablosu:** README'ye ve benchmark sayfasına aynı betikle yazılıyor.
- **İç belgeler:** `docs/plan` ve `docs/reports` sitede yer almıyor.
- **README:** Bir giriş sayfasına indi (318 → 99 satır): kurulum, hızlı başlangıç, sonuç tablosu,
  bağlantılar. Tüm bağlantılar mutlak, çünkü PyPI README'yi depo olmadan gösteriyor. Bir test bunu
  koruyor.
- **Testler:** CLI referansının güncelliği ve kullanıcıya görünen sayfalarda Faz/Hafta geçmemesi.
- **Build:** `mkdocs build --strict` uyarısız, 0,7 sn.

## 6. Lisans ve topluluk

- **NOTICE:** Kodu ya da metni alınan projelere atıf:
  - CLM (Apache-2.0): head mimarisi;
  - ToolRet (Apache-2.0): görev tablosu;
  - MCP-Zero (MIT): iki prompt; lisans metni aynen eklendi.
- **THIRD_PARTY_NOTICES.md:** `scripts/third_party.py`, `uv.lock`'tan üretiyor. İçinde:
  - imajlardaki 32 paket ve her birinin kendi metadata'sındaki lisansı;
  - diğer ekstralar; psycopg (LGPL-3.0) opsiyonel;
  - modeller ve head'lerin eğitim verisi uyarısı;
  - taban imajlar (Debian, Node.js, uv; tek imaj için CUDA koşulları; NGC tabanlı build'ler
    yayımlanmıyor).

  İmajdaki paketlerin hepsi serbest lisanslı. Bir test dosyayı güncel tutuyor.
- **Topluluk dosyaları:**
  - Contributor Covenant 2.1;
  - SECURITY.md: GitHub özel bildirimi, varsayılanların güvenlik özeti;
  - CONTRIBUTING.md;
  - issue formları (hata, özellik, benchmark sonucu) ve PR şablonu;
  - CHANGELOG.md (0.1.0, yayımlanmadı).
- **Yer tutucular:** İletişim adresleri `TODO(launch)` ile işaretli. `scripts/release_check.py`
  bunları yakalıyor, aşağıdaki listeye bakın.

## 7. CI ve yayın iş akışları

- **`ci.yml`** (dört job; push edilene kadar çalışmıyor):
  - test: Python 3.11–3.13, `uv sync --locked` (bayat bir `uv.lock` burada düşer),
    `ruff format --check`;
  - docs: strict build ve üretilen sayfaların güncelliği;
  - package: wheel'in tek başına duman testi; `serve` `[mcp]`'siz kurulum satırını vermeli;
  - image: ince imajın push'suz build'i ve sahte embedding'le duman testi.
- **`release.yml`** (`v*` etiketiyle):
  - yayın kontrolü;
  - PyPI trusted publishing (depoda token saklanmıyor);
  - iki imaj GHCR'a, amd64 + arm64, sha256'sı kontrol edilmiş head'lerle;
  - CHANGELOG'dan GitHub release.
- **`docs.yml`:** Pages'e elle dağıtım.
- **Güvenceler:** Yayın ve docs iş akışları, `TOOLRANK_RELEASE` / `TOOLRANK_PAGES` depo
  değişkenleri açılmadıkça hiçbir şey yapmıyor. PyPI ve GHCR adımları onay isteyen ortamlardan
  (`pypi`, `ghcr`) geçiyor.
- **Denetim:** `actionlint` hata bulmadı, giriş betiği `shellcheck` temiz. CI'ın paket job'ı yerelde
  birebir tekrarlandı.

## Hafta 7 kontrol listesi (`scripts/release_check.py` çıktısı)

1. CODE_OF_CONDUCT.md ve SECURITY.md'deki iletişim adresi (`TODO(launch)`).
2. Head'lerin barındırılması: `hf auth login`, ardından `scripts/publish_heads.py --upload`. Sonra
   `HEADS_URL` doldurulacak; model kartındaki "barındırılmıyor" cümlesini betik değiştiriyor.
3. Sürüm `0.1.0`, CHANGELOG'da tarihli `## [0.1.0] - YYYY-MM-DD` girdisi.
4. GitHub'da depo (`yasinyaman/toolrank`), `pypi` / `ghcr` ortamları (onaylı) ve PyPI'da trusted
   publisher.
5. Depo değişkenleri `TOOLRANK_RELEASE`, `TOOLRANK_PAGES`.

Hepsi 19:00'dan sonra ve her adım ayrı onayla.

## Komutlar

```bash
# Mac, repo kökü
uv run ruff format --check src tests scripts examples && uv run ruff check src tests scripts examples && uv run pytest
uv build --out-dir dist/pypi && uvx twine check --strict dist/pypi/*
uv run --group docs mkdocs build --strict
uv run python scripts/release_check.py            # Hafta 7'nin listesi

# GB10, ~/toolrank: FP8 (profil fp8, 8094) ve karşılaştırma
docker compose -f deploy/spark/compose.yaml --profile fp8 up -d qwen3-embedding-8b-fp8
EMB_URL=http://127.0.0.1:8094/v1 EMB_MODEL=qwen3-emb-fp8 TAG=fp8 ROWS="qwen3emb heads" \
  PYTHONUNBUFFERED=1 nohup bash scripts/readme_results.sh > data/logs/readme_fp8.log 2>&1 &
uv run python scripts/fp8_agreement.py --a http://127.0.0.1:8091/v1=qwen3-emb --b http://127.0.0.1:8094/v1=qwen3-emb-fp8 \
  --truncate 8192 --tool-format documentation --query-format instruct_query --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz
uv run python scripts/latency.py --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
  --emb-url http://127.0.0.1:8094/v1 --emb-model qwen3-emb-fp8 --truncate 8192 --tool-format documentation --query-format instruct_query

# GB10: resmi vLLM v0.30.0 (bf16, 8095) ve MCP setlerinde parite
docker run -d --name toolrank-vllm030-test --gpus all --ipc host -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 8095:8000 vllm/vllm-openai:v0.30.0 Qwen/Qwen3-Embedding-8B --served-model-name qwen3-emb --runner pooling \
  --max-model-len 8192 --dtype bfloat16 --gpu-memory-utilization 0.2 --no-enable-chunked-prefill --max-num-batched-tokens 8192
toolrank eval --data data/mcp_zero_server --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
  --emb-url http://127.0.0.1:8095/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation \
  --query-format instruct_query --with-inst --ks 1,5,10,20 --out results/readme_v030_mcp_zero_server_heads.json

# GB10: imajlar ve duman testi
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t toolrank:dev .
docker build -f deploy/docker/Dockerfile.vllm --build-context heads=dist/heads -t toolrank-vllm:dev .
uv run python scripts/container_smoke.py toolrank:dev --version 0.1.0.dev0
```

## Ortam

- GB10:
  - Qwen3-Embedding-8B bf16, NGC vLLM 0.13 (8091); FP8 kopyası aynı imajla (8094);
  - vLLM resmi imajı v0.30.0 (CUDA 13.0, arm64; 8095, test);
  - Docker 29.6, buildx 0.35 (arm64).
- Mac: Python 3.13, uv 0.11.14. Docker daemon kapalıydı, bu yüzden amd64 imaj yerelde denenmedi;
  CI'daki image job'ı ve release iş akışı amd64'ü build ediyor.
- Commit'ler: `0df128c` … `462ae5f`, ardından bu rapor.

## Sapmalar ve açıklamalar

- **Yayın Hafta 7'de.** Kararınızla PyPI, GHCR ve Pages yayını Hafta 7'ye kaldı. 2. kutu işaretlendi,
  yayın ayrı bir madde olarak Hafta 7'ye eklendi.
- **Plan ajanının incelemesinden eklenenler:**
  - `0.0.0.0`'da loopback adları;
  - FP8/bf16 için ayrı sunum adları;
  - compose'larda ayrı proje adları;
  - imajlarda tini ve önceden derlenmiş bytecode;
  - README'de mutlak bağlantılar;
  - eksik ekstralar için kurulum satırları;
  - yayın kontrolüne CHANGELOG ve model kartı;
  - CI'da Python 3.13 ve kilitli sürümler;
  - yayın iş akışlarının depo değişkenleriyle kapalı tutulması.
- **İmaj adları:** İki imaj ayrı adlarla yayımlanacak (`toolrank`, `toolrank-vllm`); plandaki
  `:<sürüm>-vllm` etiketi yerine.
- **NGC imajı:** NGC tabanlı tek imaj yalnız yerel kullanım içindir; dağıtım koşulları izin vermiyor.
- **amd64:** amd64 imajlar yalnız CI'da build edilecek. Mac'te Docker daemon kapalıydı.

## Sonraki hafta

Faz 1 Hafta 7, lansman:

- yukarıdaki kontrol listesi;
- ilk push ve CI;
- PyPI, GHCR ve Pages yayını (19:00 sonrası, adım adım onayla);
- v0.1.0 GitHub release;
- duyurular (Show HN, r/LocalLLaMA, MCP Discord, X/LinkedIn);
- ToolRet leaderboard gönderimi;
- geri bildirim görüşmeleri.
