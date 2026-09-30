# Faz 2 — Hafta 1 raporu (30 Eylül 2026): günlükten öğrenme

## Sonuç (tek cümle)

`toolrank learn` çalışıyor: kullanım günlüğü, istek metni olmadan, vektör üzerinden eğitim çiftine
dönüyor; gerçek günlük (`data/w3`, bu haftanın e2e koşuları) henüz 6 çift veriyor, yani ilk gerçek
eğitim trafik biriktikten sonra.

## Tablo

Sentetik uçtan uca test (`tests/test_learn.py`; 30 tool, 60 istek, isteğin vektörü tool'unun
döndürülmüş ve gürültülü kopyası, 16 boyut, taze skip head'ler):

| Adım | log.Recall@5 (12 dev isteği) | Karar |
| --- | ---: | --- |
| Başlangıç (birim head'ler) | düşük (döndürme öğrenilmemiş) | — |
| En iyi epoch | başlangıcı geçiyor | `published`; `.npz` `NumpyHeads` ile yükleniyor |

Gerçek günlük, `toolrank learn --data data/w3 --dry-run`:

| | |
| --- | ---: |
| Olay | 320 (137 arama, 183 çağrı) |
| Vektörsüz arama (v1/v2 şema ya da anahtar sözcük cevabı) | 76 |
| Çağrısız arama | 42 |
| Bağlanan çağrı | 79 (8 bağsız, 2 hiçbir şey söylemeyen) |
| İstek (tekil `emb_hmac`) | 8 |
| Kullanılabilir çift | 6 (0 zayıf pozitif, 4 negatif) |
| Karar | `not enough pairs` (eşik 20) |

E2E betikleri aynı birkaç isteği tekrarladığı için 137 arama 8 isteğe iniyor; öğrenme döngüsünün
değeri gerçek, çeşitli trafikle ölçülecek.

## Komutlar

```bash
uv run toolrank learn --data data/w3 --dry-run            # madencilik ve vektör eşleme raporu, torch'suz
uv run toolrank learn --data data/w3 --dev data/livemcpbench_server   # trafik birikince: eğitim + koruma seti
uv run pytest -q tests/test_learn.py tests/test_usage.py
```

## Ortam

- Mac; günlük `data/w3/usage` (v1–v3 karışık; yalnız v3 kullanılabiliyor), önbellek `data/w3/cache`.

## Sapmalar ve açıklamalar

- Takvimden erken: Faz 2, planın 4 Ocak yerine 30 Eylül'de başladı (Faz 1 erken bitti).
- Faz 2'nin "günlük şeması v1" kutusu Faz 1'in v3 şemasıyla karşılandı; yeni şema açılmadı.
- Dev seti günlüğün zaman bölmesi (en yeni %20); Faz 0'ın "eğitim çiftlerinde recall yanıltır" dersi
  nedeniyle ayrıca `--dev` benchmark koruması var. Küçük günlüklerde Recall@5 kaba adımlarla oynar;
  rapor bunu uyarı olarak yazıyor.
- Servisin çalışırken head değiştirmesi ve %10 A/B (Hafta 3–4) henüz yok: yayınlanan `.npz`
  `TOOLRANK_HEADS` ile veriliyor.

## Sonraki hafta

- Hafta 3–4: `learn`'ün gecelik koşusu, servise çalışırken head yüklemek, A/B ve otomatik geri alma,
  unutmaya karşı genel replay karışımı.
- Gerçek trafik: kendi Claude Code / Claude Desktop kullanımını `data/w3` üzerinden `serve` ile
  günlüklemek; ilk anlamlı `learn` koşusu birkaç yüz istekte.
