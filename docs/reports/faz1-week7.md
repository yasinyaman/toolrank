# Faz 1 — Hafta 7 raporu (30 Eylül 2026): yayın

## Sonuç (tek cümle)

toolrank 0.1.0 yayında: head'ler Hugging Face'te, paket PyPI'da, imaj GHCR'da, kaynak ve docs
GitHub'da; hepsi yalnız herkese açık kaynaklardan kurulup uçtan uca denendi. Duyurular, ToolRet
issue'su ve kapı ölçümü (lansman + 6 hafta) önde.

## Yayımlananlar

| | Adres | Doğrulama |
| --- | --- | --- |
| Head'ler | `huggingface.co/yasinyaman/toolrank-heads-qwen3-emb-8b`, `v0.1` etiketi | sha256 `f3c10125…72f0`; temiz cache'e `toolrank heads pull` aynı dosyayı indirdi |
| Paket | `pypi.org/project/toolrank/` 0.1.0 (wheel 158 KB `01086bd3…`, sdist 209 KB `8cdb996a…`) | temiz 3.12 venv'ine `pip install "toolrank[mcp]==0.1.0"` |
| İmaj | `ghcr.io/yasinyaman/toolrank:0.1.0` (`0.1`, `latest`), amd64 + arm64, index `sha256:d8905879…7fa94` | GB10'da `docker pull` (arm64, 504 MB, head'ler içinde) + `container_smoke.py`: 8/8 |
| Kaynak | `github.com/yasinyaman/toolrank`, tek commit `c8102f2`, release `v0.1.0` | CI ilk koşuda yeşil (6 iş); release iş akışı 7 iş, 12 dakika (onaylar dahil) |
| Docs | `yaman.dev/toolrank/` (`yasinyaman.github.io/toolrank/` oraya 301) | ana sayfa, quickstart, CLI referansı, Docker rehberi 200 |

Kurulu paketle uçtan uca (embedding'ler GB10'daki Qwen3-Embedding-8B'den, head'ler Hub'dan):
`uvx mcp-server-time` ingest, `toolrank search` (`time/get_current_time` 0.60 ilk sırada),
`toolrank serve` ile REST arama 16 ms, `/v1/call` Tokyo saatini döndürdü, MCP initialize 200,
token'sız istek 401, kapanışta çocuk süreç yok.

## Yayın öncesi (aynı gün)

- **Güvenlik incelemesi** (tüm kod): iki açık kapandı — `/` ile başlamayan bir spec path'i config'deki
  kimlik bilgilerini başka bir sunucuya taşıyabiliyordu (`backends.same_origin` artık gönderilen
  URL'i base URL'le karşılaştırıyor); `.pt` dosyaları `weights_only` olmadan yükleniyordu.
- **İki kod incelemesi:** 14 + 8 bulgu, hepsi giderildi. Çoğu birleşik imajın sarmalayıcısında
  (`as-toolrank.sh`): toolrank artık komutun yazdığı dizinin sahibi olarak, root grubuna girmeden,
  `/home` altında bir evle çalışıyor; yalnız root'un `/data`'daki dosyaları el değiştiriyor.
- Yayın akşamı CI'ı düşürecek iki hata yakalandı: `astral-sh/setup-uv@v10` diye bir etiket yokmuş
  (tam sürümler var); `container_smoke.py`'nin geçici dizini konteyner kullanıcısına kapalıydı.
- Kullanım günlüğü şema v3: `emb_hmac`, istekle gelen talimat yalnız özet olarak, bilinmeyen tool
  adı 200 karakterde.
- Tedarik zinciri: action'lar commit'e, taban imajlar digest'e sabitli; imajlar lock'un hash'leriyle
  kuruluyor; Dependabot ilk PR'ını özel depodayken açtı (`#1`, uv imajı; aynı akşam birleşti).
- Doğrulama: GB10'da tam test paketi (195 test, atlanan yok), README tablosunun 12 koşusu önbellekten
  yeniden (docs/results ile alan alan birebir), 29 torch checkpoint'i `weights_only` ile, birleşik
  imaj gerçek modelle iki kurulumda, Mac'ten üç e2e betiği.

## Komutlar

```bash
# Mac, repo kökü; her herkese açık adım ayrı onayla, 19:00'dan sonra
uv run --with huggingface_hub python scripts/publish_heads.py --repo yasinyaman/toolrank-heads-qwen3-emb-8b --upload
uv run python scripts/release_check.py --tag v0.1.0            # "ready to release 0.1.0"
git branch -m main local-history && git checkout --orphan main
git -c user.email=99225310+yasinyaman@users.noreply.github.com commit -m "faz-1/w7: toolrank 0.1.0"
gh repo create yasinyaman/toolrank --private --source . --remote origin && git push origin main
gh repo edit yasinyaman/toolrank --visibility public --accept-visibility-change-consequences   # bakımcı
gh api -X POST repos/yasinyaman/toolrank/rulesets --input main.json          # main: silme ve force-push yok
gh api -X PUT repos/yasinyaman/toolrank/environments/pypi --input env.json    # gözden geçiren, yalnız v* etiketleri
gh workflow run docs.yml -R yasinyaman/toolrank
git tag -a v0.1.0 -m "toolrank 0.1.0" && git push origin v0.1.0            # ghcr ve pypi onayları Actions'ta
# doğrulama, yalnız herkese açık kaynaklardan
uv venv /tmp/pub && uv pip install --python /tmp/pub/bin/python "toolrank[mcp]==0.1.0" && /tmp/pub/bin/toolrank heads pull
ssh gb10 'docker pull ghcr.io/yasinyaman/toolrank:0.1.0 && cd ~/toolrank && uv run python scripts/container_smoke.py ghcr.io/yasinyaman/toolrank:0.1.0 --version 0.1.0'
uv run python scripts/launch_metrics.py --since 2026-09-30 --discussions --out results/launch_metrics.json
```

## Ortam

- Mac (git, yayın adımları, e2e betikleri); GB10: Qwen3-Embedding-8B bf16 (8091), `docker` 29.
- GitHub Actions: `ubuntu-latest`; imaj `docker/build-push-action` ile iki platform (QEMU).
- PyPI trusted publisher (pending publisher, `release.yml`, ortam `pypi`); GHCR paketi bakımcı
  tarafından herkese açık yapıldı.

## Sapmalar ve açıklamalar

- Yayın plandaki "5 Ekim'den sonra" yerine 30 Eylül akşamı yapıldı; karar bakımcının. Duyurular
  başka bir akşama kaldı.
- Pages adresi `yaman.dev/toolrank/` oldu: hesabın kullanıcı sitesinde özel alan adı var, proje
  sayfaları onun altına düşüyor. `github.io` adresi 301 ile yönleniyor; `site_url` ve bağlantılar
  sonra çevrilecek (backlog).
- Depoyu herkese açık yapan komut Claude Code'un izin katmanına takıldı; o adımı bakımcı çalıştırdı.
  Kural setleri, ortamlar ve Pages ücretsiz planda ancak herkese açık depoda kurulabiliyor.
- Açık geçmişin ilk commit'i kitteki mesajla, ortak yazar satırı olmadan atıldı.

## Sonraki hafta

- Duyurular (Show HN, r/LocalLLaMA, MCP Discord, X, LinkedIn) ve ToolRet issue'su; ilk hafta
  cevapları ve görüşme davetleri (`lansman-kiti.md`).
- `site_url` → `yaman.dev`, T+7 ölçümü. (Dependabot'un ilk PR'ı, uv imajı 0.11.33 → 0.12.21, aynı akşam
  birleştirildi; CI iki kez yeşil. Yayımlanan imaj etiketten geldiği için 0.11.33 ile kaldı.)
