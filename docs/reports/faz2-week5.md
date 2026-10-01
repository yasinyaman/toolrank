# Faz 2 — Hafta 5 raporu (2 Ekim 2026): sunucu yönlendirmesi, birlikte-kullanım, "tool yok" kapısı

## Sonuç (tek cümle)

Hiyerarşik routing'in MCP-Zero kalıbı (önce sunucuyu seç, sonra yalnız onun tool'larında ara) her
sette puan kaybettiriyor; yumuşak hali — tool skoruna sunucusunun skorundan 0,2 pay eklemek —
MCP-Zero top-1'i 79,87 → 80,98'e çıkarıyor ve `--server-weight` olarak eklendi; günlükten çıkan
birlikte-kullanım tablosu (`serve --co-use`) listelerin %5'ine 0,05 tool ekleyip tamlığı +0,6 puan
artırıyor; "bu istek için tool yok" kararı ise skor eşiğiyle güvenilir verilemiyor (AUROC 0,88 /
0,76, eşik katalogdan kataloğa taşınmıyor) — üçü de varsayılan olarak kapalı.

## Ölçüler, birimler ve yön

- `P@1 ↑ %` (MCP-Zero'nun top-1 doğruluğu), `Recall@5 ↑ %`, `NDCG@10 ↑ %` (micro; ToolRet'te ayrıca
  cat-macro): `toolrank eval` protokolü, w/ inst, v0.1 head'leri (`toolrank-heads-qwen3-emb-8b-v0.1`).
- `K@cut ↓`, `Recall@cut ↑ %`, `Comprehensiveness@cut ↑ %`: uyarlanabilir K'nın (margin 0,2, en çok 10)
  döndürdüğü listenin uzunluğu, geri çağırması ve **tamlığı** (isteğin bütün altın tool'ları listede mi).
- **Doğru sunucu ilk sırada ↑ %**: sunucu skoruna göre en iyi sunucu, isteğin altın tool'larından
  birinin sunucusu mu.
- **Kapı**: "yanıtlanabilir" = istek tüm katalogla; "yanıtlanamaz" = aynı istek, altın tool'larının
  sunucuları katalogdan çıkarılmış. Ölçü en iyi kosinüs; `AUROC ↑` ikisini ayırma gücü; bir eşikte
  "geri çevrilen yanıtlanabilir ↓ %" ve "yakalanan yanıtlanamaz ↑ %".

## Tablo 1 — sunucu → tool yönlendirmesi (bir kez sırala, çok kuralla yeniden puanla)

Sunucu = tool'un `category`'si. Sunucu skoru: **özet** (sunucu adı + tool adları tek metin olarak
gömülü), **merkez** (sunucunun tool vektörlerinin ortalaması), **ilk 3** (sunucunun en iyi üç tool
skorunun ortalaması). Kural: **ekle W** (tool + W·sunucu), **çarpım** (MCP-Zero'nun
`(sunucu·tool)·max(sunucu, tool)` formülü), **sert M** (yalnız en iyi M sunucunun tool'ları).

MCP-Zero `_server` (2.792 istek, 2.792 tool, 293 sunucu):

| Sunucu skoru | Kural | P@1 ↑ % | Recall@5 ↑ % | NDCG@10 ↑ % | Doğru sunucu ilk sırada ↑ % |
| --- | --- | ---: | ---: | ---: | ---: |
| — | yalnız tool skoru | 79,87 | 94,20 | 88,53 | |
| özet | ekle 0,1 | 80,66 | 94,42 | 89,03 | 73,21 |
| özet | **ekle 0,2** | **80,98** | 94,38 | **89,28** | 73,21 |
| özet | ekle 0,3 | 80,66 | 94,49 | 89,27 | 73,21 |
| özet | ekle 0,5 | 80,05 | 94,44 | 89,02 | 73,21 |
| özet | çarpım | 79,12 | 94,35 | 88,59 | 73,21 |
| özet | sert 1 / 3 / 10 | 63,07 / 74,46 / 78,37 | 70,83 / 86,72 / 92,05 | 67,95 / 82,07 / 86,67 | 73,21 |
| merkez | ekle 0,3 | 80,52 | 94,54 | 89,14 | 80,55 |
| merkez | çarpım | 79,51 | 94,24 | 88,49 | 80,55 |
| merkez | sert 1 / 3 / 10 | 69,91 / 78,12 / 79,44 | 78,39 / 91,09 / 93,70 | 75,20 / 86,14 / 88,09 | 80,55 |
| ilk 3 | ekle 0,3 | 80,01 | 94,24 | 88,80 | 86,28 |
| ilk 3 | sert 1 / 3 / 10 | 75,47 / 79,41 / 79,91 | 84,12 / 92,70 / 94,13 | 80,84 / 87,58 / 88,54 | 86,28 |

Diğer setlerde aynı kurallar (özet skoru; tam tablolar komutların çıktısında):

| Set | Yalnız tool (P@1 / NDCG@10) | Ekle 0,2 | Ekle 0,3 | Çarpım | Sert 3 (ToolRet: sert 1) |
| --- | --- | --- | --- | --- | --- |
| LiveMCPBench `_server` (94 istek, 69 sunucu) | 48,94 / 53,95 | 50,00 / 55,06 | 48,94 / 54,96 | 51,06 / 55,21 | 46,81 / 49,61 |
| MCP-Zero, tool metninde sunucu adı yok | 72,10 / 83,80 | 74,28 / 85,46 | 74,61 / 85,85 | 73,50 / 85,47 | 68,55 / 78,80 |
| LiveMCPBench, sunucu adı yok | 45,74 / 52,84 | 48,94 / 54,32 | 47,87 / 54,49 | 51,06 / 55,27 | 43,62 / 48,44 |
| MCP-Zero `_server`, head'siz (Qwen3-Embedding-8B) | 78,19 / 87,21 | 79,48 / 88,24 | 79,44 / 88,43 | 79,12 / 88,33 | 73,96 / 82,14 |
| ToolRet (44.453 tool, 3 kategori "sunucu") | 45,75 / 54,01 | 45,56 / 53,91 | 45,25 / 53,78 | 23,63 / 28,77 | 23,98 / 29,08 |

## Tablo 2 — ürün yolunda: `toolrank eval --server-weight W --cut-margin 0.2`

| Set | W | P@1 ↑ % | Recall@5 ↑ % | NDCG@10 ↑ % (micro / cat-macro) | K@cut ↓ | Recall@cut ↑ % | Comprehensiveness@cut ↑ % |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MCP-Zero `_server` | 0 | 79,87 | 94,20 | 88,53 | 5,88 | 95,82 | 95,70 |
| | 0,2 | 80,98 | 94,38 | 89,28 | 5,52 | 96,14 | 96,02 |
| | 0,3 | 80,66 | 94,49 | 89,27 | 5,39 | 96,36 | 96,24 |
| LiveMCPBench `_server` | 0 | — | 53,03 | 53,95 / 53,52 | 7,45 | 59,27 | 34,04 |
| | 0,2 | — | 52,15 | 55,06 / 54,69 | 7,17 | 60,12 | 36,17 |
| | 0,3 | — | 52,41 | 54,96 / 54,81 | 7,11 | 60,12 | 36,17 |
| ToolRet | 0 | — | 57,39 | 54,03 / 47,13 | 8,29 | 64,96 | 54,34 |
| | 0,2 | — | 57,32 | 53,93 / 46,36 | 8,29 | 64,86 | 54,37 |
| | 0,3 | — | 57,23 | 53,79 / 45,74 | 8,29 | 64,81 | 54,39 |

## Tablo 3 — "tool yok" kapısı: en iyi kosinüs üzerinde eşik

| | MCP-Zero `_server` | LiveMCPBench `_server` |
| --- | ---: | ---: |
| Yanıtlanabilir, ortalama en iyi skor | 0,708 | 0,505 |
| Yanıtlanamaz (altın sunucular çıkarılmış), ortalama | 0,535 | 0,377 |
| AUROC ↑ | 0,882 | 0,761 |

| Eşik | MCP-Zero: geri çevrilen yanıtlanabilir ↓ % | yakalanan yanıtlanamaz ↑ % | LiveMCPBench: geri çevrilen ↓ % | yakalanan ↑ % |
| ---: | ---: | ---: | ---: | ---: |
| 0,30 | 0,00 | 2,26 | 8,51 | 30,85 |
| 0,40 | 0,14 | 12,50 | 19,15 | 57,45 |
| 0,45 | 0,82 | 23,10 | 29,79 | 73,40 |
| 0,50 | 2,08 | 37,64 | 44,68 | 85,11 |
| 0,55 | 5,27 | 54,01 | 60,64 | 89,36 |
| 0,60 | 12,25 | 70,09 | 78,72 | 94,68 |

Simüle günlüklerde (Hafta 4) çağrıya giden ve gitmeyen aramaların en iyi skoru da örtüşüyor: çağrıya
gidenlerin %99'unu koruyan eşik (ToolRet 0,417, MCP-Zero 0,481) gitmeyenlerin %5 / %14'ünü boşaltıyor.

## Tablo 4 — birlikte-kullanım (ToolRet simüle günlüğü, 5.573 istek; ayrılmış 2.388 sorgu, 1.064'ü çok tool'lu)

Tablo: A tool'uyla aynı istekten sonra en az `count` istekte ve A'nın çağrıldığı isteklerin en az
`p` payında çağrılmış B'ler; kesilmiş listeye, içindeki tool'ların en çok `N` ortağı eklenir.

| Liste | K ↓ | Recall ↑ % | Precision ↑ % | Comprehensiveness ↑ % | Çok tool'lu isteklerde ↑ % | Değişen liste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uyarlanabilir K (margin 0,2, en çok 10) | 8,33 | 64,81 | 18,53 | 54,15 | 31,48 | |
| uyarlanabilir K, en çok 12 | 9,65 | 66,79 | 17,79 | 56,20 | 33,83 | |
| uyarlanabilir K, en çok 15 | 11,50 | 69,13 | 17,07 | 58,58 | 36,94 | |
| + birlikte-kullanım (count ≥ 2, p ≥ 0,3, en çok 2) | 8,39 | 65,16 | 18,36 | 54,73 | 32,80 | 126 |
| **+ birlikte-kullanım (count ≥ 2, p ≥ 0,5, en çok 2)** | **8,38** | 65,16 | 18,39 | **54,73** | **32,80** | 111 |
| + birlikte-kullanım (count ≥ 2, p ≥ 0,5, en çok 1) | 8,37 | 65,00 | 18,38 | 54,36 | 31,95 | 111 |
| + birlikte-kullanım (count ≥ 2, p ≥ 0,7, en çok 2) | 8,35 | 64,98 | 18,47 | 54,44 | 32,14 | 54 |
| + birlikte-kullanım (count ≥ 3, p ≥ 0,5, en çok 2) | 8,36 | 65,05 | 18,42 | 54,61 | 32,52 | 64 |

| Aynı ayar (2 / 0,5 / 2), başka koşullar | K ↓ | Comprehensiveness ↑ % | Çok tool'lu ↑ % | Değişen liste |
| --- | ---: | ---: | ---: | ---: |
| Günlükten öğrenilmiş head'lerle, eklemesiz | 7,92 | 55,61 | 32,14 | |
| Günlükten öğrenilmiş head'lerle, + birlikte-kullanım | 7,97 | 56,24 | 33,55 | 117 |
| Yeni tool istekleri (1.061), eklemesiz → ekli | 7,85 → 7,87 | 71,63 → 71,63 | 45,39 → 45,39 | 13 |
| Servis yolunda (`learn_sim.py traffic … --co-use 2`), 0 → 2 | 8,33 → 8,38 | 54,15 → 54,73 | | 111 |

## Komutlar

```bash
# GB10, ~/toolrank; bütün vektörler önbellekte
EMB="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation \
  --query-format instruct_query --with-inst"
E="--scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz $EMB"

# Tablo 1 ve 3 (--gate): mcp_zero_server, livemcpbench_server; sunucu adsız: mcp_zero, livemcpbench
uv run python scripts/routing_sweep.py --gate -- --data data/mcp_zero_server $E
uv run python scripts/routing_sweep.py --servers centroid,summary --weights 0.1,0.2,0.3 --hard 1 -- --data data/toolret $E
uv run python scripts/routing_sweep.py --servers centroid,summary --weights 0.1,0.2,0.3 --hard 3 -- \
  --data data/mcp_zero_server --scorer dense $EMB                              # head'siz satır

# Tablo 2: W = 0, 0.2, 0.3; MCP-Zero'da --ks 1,5,10,20
uv run toolrank eval --data data/mcp_zero_server $E --ks 1,5,10,20 --cut-margin 0.2 --server-weight 0.2 \
  --out results/route_mcp_zero_server_w0.2.json

# Tablo 4: Hafta 4'ün simüle günlüğü (bash scripts/learn_sim.sh) üzerinde
uv run python scripts/couse_sweep.py --log data/sim/toolret/runs/n0/usage --min-count 2,3 --min-p 0.3,0.5,0.7 -- \
  --data data/sim/toolret/heldout $E --cut-margin 0.2
# öğrenilmiş head'ler: --clm-ckpt data/sim/toolret/runs/n0/learned/default.npz; yeni tool'lar: --data …/heldout_new
# servis yolu: günlüğün bir kopyasında (runs/n0_couse), ayrılmış sorgular
uv run python scripts/learn_sim.py traffic --queries data/sim/toolret/heldout/queries.jsonl -- \
  --data data/sim/toolret/runs/n0_couse --emb-url http://127.0.0.1:8091/v1 --cache-dir .cache/toolrank --co-use 2
```

## Ortam

- GB10; Qwen3-Embedding-8B bf16 (8091), önbellek sıcak; yeni gömülen tek şey sunucu özetleri (set
  başına sunucu sayısı kadar kısa metin). Kod: `e467615` + bu haftanın değişiklikleri.
- Yönlendirmeli sıralamanın gecikmesi değişmiyor (MCP-Zero p50 1,8–2,8 ms, batch 64; ToolRet 2,5–3,3 ms):
  indeksin ilk 100 tool'u yeniden puanlanıyor.

## Sapmalar ve açıklamalar

- **Plandaki "MCP-Zero kalıbı" tutmadı.** Doğru sunucu ilk sırada yalnız %68–86 çıkıyor (özet %73,
  merkez %81, ilk 3 %86; LiveMCPBench'te %68–70), bu yüzden sunucu seçip yalnız orada aramak
  kaybettiriyor (sert 1–3: P@1'de 0 … −17, Recall@5'te −1,5 … −23 puan). MCP-Zero'nun çarpım formülü
  tutarsız: MCP-Zero'da −0,4 … −0,8 P@1, LiveMCPBench'te +2 (94 istek), ToolRet'te çöküyor (−22).
  Her sette kazandıran tek biçim toplama: sunucu oyu sıralamayı itiyor ama kimseyi elemiyor.
- **Kazanç küçük, çünkü sunucu adı zaten tool metninde.** Faz 0'da metne sunucu adını eklemek +8
  puan vermişti; kalan pay +1,1 P@1 (MCP-Zero) ve +1,1 NDCG@10 (LiveMCPBench, 94 istek: bir istek ≈ 1
  puan, zayıf kanıt). Sunucu adı metinde yokken aynı kural +2,2 … +2,5 P@1 veriyor.
- **Neden varsayılan değil:** "sunucu"ları üç dev kategori olan ToolRet'te micro −0,1 ama cat-macro
  −0,8 (W 0,3'te −1,4). Çok sunuculu kataloglar için `--server-weight 0.2` öneri; gerçek kataloglarda
  (`data/w3`: 5 kaynak, biri 1.231 tool) ölçüm yok.
- **Özet mi merkez mi:** ikisi de çalışıyor; özet MCP-Zero'da biraz daha iyi (80,98 / 80,52) ve
  indeksten vektör geri okumayı gerektirmiyor (FAISS, pgvector), o yüzden ürüne o girdi. Özet metni
  4.000 karakterde kesiliyor (1.000 tool'luk bir sunucunun ilk birkaç yüz tool adı).
- **Kapı:** tek bir eşik iki sette aynı işi görmüyor (yanıtlanabilir ortalama 0,71'e karşı 0,51;
  LiveMCPBench istekleri çok adımlı görevler). Yanıtlanabilir isteklerin %1'inden azını geri çeviren
  eşik MCP-Zero'da yanıtlanamazların %23'ünü yakalıyor, LiveMCPBench'te böyle bir eşik yok. Bu yüzden
  varsayılan eşik yok; `--cut-threshold T --cut-min 0` ile açılıyor, boş sonuçta ajan "uygun tool yok"
  notunu görüyor, günlük o aramanın göstereceklerini tutmaya devam ediyor (kalibrasyon için).
  "Yanıtlanamaz" tanımı sert: sunucusu çıkarılan tool'un bir benzeri başka sunucuda kalmış olabilir.
  İsteği adaylarla birlikte okuyan ikinci aşama (`docs/reports/faz2-jev.md`) bu karar için daha doğru
  yer; backlog'a yazıldı.
- **Birlikte-kullanım küçük ama ucuz:** eklenen tool başına tamlık kazancı, listeyi uzatmanın yedi
  katı (0,05 tool için +0,58; en çok 12'ye uzatmak 1,32 tool için +2,05). Mutlak kazanç küçük çünkü
  44 bin tool'luk katalogda 5.573 istek yalnız 164 tool'a ortak buldurdu; aynı tool çiftlerinin tekrar
  ettiği gerçek trafikte payın büyümesi beklenir, ölçülmedi. Öğrenilmiş head'lerle kazanç aynı (+0,63):
  ikisi birbirinin yerine geçmiyor. Günlükte görülmemiş tool'lara etkisi yok, zararı da yok.
- **Günlük şeması:** arama olayına isteğe bağlı `added` alanı geldi (birlikte-kullanımla eklenenler);
  `shown` sıralamadan gelenlerin sayısı olarak kaldı, `learn` ikisini de "gösterildi" sayıyor. Sürüm
  numarası değişmedi (v3; alan yoksa eski davranış).
- Simülasyon notu Hafta 4'teki gibi: ajan kusursuz, istekler benchmark sorguları.

## Sonraki hafta

- Hafta 6: tenant izolasyonu (indeks, head, günlük; birlikte-kullanım tablosu şu an bütün anahtarları
  birlikte sayıyor), Helm chart, Prometheus metrikleri.
- Gerçek trafik biriktikçe: `--server-weight` ve `--co-use`'un `data/w3` üzerinde A/B'si.
