# Faz 1 — Kapı raporu (taslak)

**Durum.** Taslak. Faz 1'in işi Hafta 7'yle bitiyor. Çıkış kriteri lansmandan 6 hafta sonra ölçülüyor.

| | |
| --- | --- |
| Lansman tarihi | 30 Eylül 2026, 21:10 (v0.1.0 PyPI'da) |
| Ölçüm günü | lansman + 42 gün: 11 Kasım 2026 |
| Sonuç | ölçüm günü yazılıyor |

## Sonuç (tek cümle)

(ölçüm günü)

## Çıkış kriterleri ve ölçüm

"Dış" demek: maintainer değil, bot değil. Sayılar `scripts/launch_metrics.py`'den geliyor, yazılı geri
bildirimler ise elle sayılıyor:

```bash
uv run python scripts/launch_metrics.py --since <lansman tarihi> --discussions --out results/launch_metrics_<tarih>.json
```

| Kriter | Eşik | Ölçüm |
| --- | ---: | --- |
| GitHub yıldızı | 300 | `stars` |
| Dış kullanıcılardan yazılı geri bildirim | 3 | Aşağıdakilerin toplamı:<br>• dışarıdan açılan `feedback` etiketli issue'lar (`feedback_issues`);<br>• dışarıdan açılan ve bir kurulumu anlatan Discussion yazıları;<br>• görüşmeden sonra kişinin kendi yazdığı ya da onayladığı özetler.<br>Her biri bağlantısıyla aşağıda listeleniyor. |
| Dış PR ya da issue tartışması | 1 | `outside_threads`: dış bir PR, ya da dışarıdan açılıp en az bir cevap almış bir issue veya Discussion |

## Ölçümler

T+0 lansman akşamı yazılıyor, sonra haftada bir.

| Tarih | Yıldız | Fork | Dış issue | Dış PR | Dış tartışma | `feedback` | PyPI (son hafta) | Head indirme (30 gün) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T+0 (30 Eyl 21:15) | 0 | 0 | 0 | 0 | 0 | 0 | — | 0 |

**Yazılı geri bildirimler** (bağlantı, tarih, bir cümlelik özet):

1. —

## Faz 1'de yapılanlar

| Hafta | Çıktı | Sayılar |
| --- | --- | --- |
| 1 | Ingestion: MCP (stdio ve streamable HTTP) ve OpenAPI 3.x; artan senkron | Resmi MCP sunucuları, GitHub REST ve Stripe'la denendi (`faz1-week1.md`) |
| 2 | `toolrank search`: kalıcı indeks (numpy, FAISS HNSW, pgvector), uyarlanabilir K, torch'suz paketlenmiş head'ler | ToolRet 54.03 / 47.13 cat-macro; LiveMCPBench 53.03; MCP-Zero 79.87 (`faz1-week2.md`) |
| 3 | `toolrank serve`: iki tool'luk MCP proxy'si, REST, kullanım günlüğü | 1.862 tool'da sıcak arama 5.2 ms. İlk açılış boş cache'le 154 sn, dolu cache'le 0.8 sn (`faz1-week3.md`) |
| 4 | Claude'un ve OpenAI'ın tool search kancaları | Claude'da üç görevin üçü doğru, API'nin kendi BM25'ine göre %37–60 daha az girdi token'ı (`faz1-week4.md`) |
| 5 | `toolrank finetune`; LangGraph, LlamaIndex, LiteLLM; README'nin sonuç tablosu | Üç framework de `data/w3`'te doğru tool'u buldu (`faz1-week5.md`) |
| 6 | Paket, Docker imajları ve compose, FP8 varsayılanı, docs sitesi, lisans ve topluluk dosyaları, CI | FP8: 53.94 / 47.27, 53.48, 79.51; batch-1 gecikmesi 99 → 55 ms (`faz1-week6.md`) |
| 7 | Yayın ve lansman | (`faz1-week7.md`) |

## Karar

Ölçüm günü yazılıyor. Kriterler tuttu mu, tutmadıysa kapsam nasıl daralıyor? Takvim kaymıyor.

## Açık kalanlar

- **ToolRet'te 50 cat-macro.** Faz 0'dan kalan eşik tutmadı; sayı 47.13. Kaldıraçlar backlog'da: LoRA,
  temiz negatifler, LLM'le zenginleştirilmiş tool metni.
- **toolrank-vllm imajı GHCR'da değil.** Ücretsiz runner'da iki mimariye sığmıyor. Her mimari kendi
  runner'ında build edilip birleştirilince yayımlanabilir.
- **GPU'suz bir yol yok.** Ollama ya da llama.cpp'deki GGUF Qwen3-Embedding-8B'nin head'lerle ölçülmesi
  bekliyor.
- **Head'lerin eğitim verisinin lisansı yok.** ToolRet-Training-20w lisans belirtmiyor; model kartında
  yazıyor.
