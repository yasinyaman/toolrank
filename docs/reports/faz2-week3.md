# Faz 2 — Hafta 3 raporu (1 Ekim 2026): çalışırken head değişimi, A/B, replay

## Sonuç (tek cümle)

Öğrenme döngüsü kapandı: `toolrank learn` adayı `DATA/heads/candidate.npz` olarak yazıyor, çalışan
sunucu onu yeniden başlamadan isteklerin bir payına veriyor, `toolrank ab` günlükten iki kolu
karşılaştırıp adayı yükseltiyor ya da geri alıyor; mekanizma gerçek sunucuda doğrulandı, kalite
iddiası için hâlâ gerçek trafik gerekiyor.

## Tablo

Gerçek sunucu, `data/w3` (1.862 tool, 5 kaynak), embedding'ler GB10'daki Qwen3-Embedding-8B'den,
aday olarak paketlenmiş head'lerin kopyası (aynı sıralama, yalnız kol değişiyor), `--candidate-share 1.0`:

| Adım | Süre ↓ | Günlükteki `arm` |
| --- | ---: | --- |
| Başlangıç (bayrakların head'leri) | arama 24 ms (ilk), 3–5 ms (sıcak) | `base` |
| `candidate.npz` dizine kondu | 1 sn içinde devrede; 1.862 tool'un yeniden izdüşümü 0,6 sn | `candidate` |
| `toolrank ab --promote` | yeniden adlandırma; sunucu 0,2 sn'de `current`'ı kurdu | `current` |

Hiçbir metin yeniden gömülmedi: tool vektörleri önbellekten geldi, yalnız head'ler yeniden uygulandı.

Birim ve uçtan uca testler (230 test, hepsi geçiyor):

| Ne | Nerede |
| --- | --- |
| `current` / `candidate` / `tenant:<ad>` / `tenant:<ad>:candidate` varyantları, yapışkan pay, dosya değişince yeniden kurma, dosya gidince düşme, bozuk dosya, `--clm-ckpt` önceliği | `tests/test_retriever.py` |
| Kol başına arama, çağrı, top-1, `mrr`; `promote` / `rollback` / `wait`; dosya taşıma; `toolrank ab` çıktısı | `tests/test_learn.py` |
| `learn`'ün adayı yazması, replay çiftlerinin karışması (endpoint'e istek yok), yargılanan adayın üstüne yazmama | `tests/test_learn.py` |
| Günlükte `arm` alanı | `tests/test_usage.py` |

## Komutlar

```bash
# Mac, repo kökü; $GB10: GB10'un Tailscale adresi (depoda tutulmuyor), hep aynı adres
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz uv run toolrank serve --data data/w3 \
  --candidate-share 1.0 --port 18791 --api-key "$KEY" --emb-url http://$GB10:8091/v1
cp dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz data/w3/heads/candidate.npz   # sunucu çalışırken
uv run toolrank ab --data data/w3 --dry-run
uv run toolrank ab --data data/w3 --promote
# her gece: dünkü adayı yargıla, yenisini öğren
toolrank ab --data data/mytools && toolrank learn --data data/mytools --replay data/toolret_train/pairs.jsonl
```

## Ortam

- Mac (sunucu), GB10 (Qwen3-Embedding-8B bf16, 8091); head'ler fp16 `.npz` (`f3c10125…`).

## Sapmalar ve açıklamalar

- **Otomatik geri alma** bir arka plan süreci değil, `toolrank ab` koşusu: karar günlükten çıkıyor ve
  dosya adlarıyla uygulanıyor; her gece çalıştırılınca "metrik iyileşmezse geri al" sağlanıyor.
- **A/B ölçüsü `mrr`** (çağrılan tool'un 1/sırasının, kolun tüm aramaları üzerinden ortalaması):
  ajan yalnız gösterilen tool'ları çağırabildiği için "ilk 5'te mi" neredeyse hep doğru çıkar ve kolları
  ayırmaz. `mrr` hem doğru tool'un yukarıda durmasını hem de aramanın bir çağrıya yol açmasını ödüllendiriyor.
- **Kişiye özel head** API anahtarının adıyla (`tenants/<ad>/`); anahtarsız kurulumda tek bir ortak
  head var.
- Karar eşiği (kol başına 100 arama, 0,01 `mrr` farkı) istatistiksel bir test değil; küçük farklarda
  aday beklemede kalıyor. Gerçek trafikle birlikte gözden geçirilecek.
- Bu haftanın sayıları mekanizmanın çalıştığını gösteriyor; öğrenilen head'lerin kazancı hâlâ
  ölçülmedi (`data/w3` günlüğü 6 çift).

## Sonraki hafta

- Gerçek trafik: kendi Claude Code / Claude Desktop kullanımını `data/w3` üzerinden günlüklemek,
  birkaç yüz istekte ilk gerçek `learn` + `ab` turu.
- Hafta 5: küme düzeyinde retrieval ve hiyerarşik routing.
