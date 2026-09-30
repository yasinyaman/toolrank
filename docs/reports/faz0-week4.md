# Faz 0 — Hafta 4 raporu (29 Eyl 2026)

## Sonuç (tek cümle)

İki held-out MCP setinde de sıralama ToolRet'tekiyle aynı çıktı. En iyisi Qwen3-Embedding-8B +
ToolRet-train'le eğitilmiş skip head'ler: MCP-Zero'da top-1 isabet 69.7'den 72.0'a,
LiveMCPBench'te Recall@5 49.0'dan 52.3'e çıkıyor. CLM iki sette de kullanılamayacak kadar zayıf
(MCP-Zero top-1 1.5, fine-tune sonrası 4.6). Head'lerden de büyük kaldıraç, tool metnine sunucu
adını eklemek: MCP-Zero'da top-1 +8 puan (BM25'te +24), LiveMCPBench'te Recall@5 +1 ile +3 arası.

## MCP-Zero

Kurulum (`toolrank data pull mcp-zero`, `src/toolrank/datasets/mcp_zero.py`):

- **Veri.** MCP-tools'ta 308 sunucu ve 2.797 tool var (`mcp_tools_with_embedding.json`, 333 MB,
  yalnız Google Drive'da, sha256 `ba517c00…0842`). 14 sunucu adı tekrar ediyor (ör. resmi ve
  topluluk `Slack`), bu yüzden tool id'si sunucunun README yolu + tool adı. Aynı sunucuda aynı adı
  taşıyan 5 tool düşüyor, geriye 2.792 tool kalıyor. Hazır embedding'ler kullanılmıyor.
- **Sorgular.** Veri setinde sorgu yok. Makalenin deneyi, her hedef tool için bir LLM'e sistem
  promptu `system_ours_mcptools.prompt` ile `I need to {tool açıklaması} with a MCP server of {sunucu
  açıklaması}.` mesajını veriyor; model `<tool_assistant> server: … tool: … </tool_assistant>`
  biçiminde bir istek döndürüyor. Burada her tool bir kez hedef oldu ve aynı promptlar kullanıldı;
  makalenin Claude-3.5-Sonnet / GPT-4.1 / Gemini-2.5-Flash'ı yerine GB10'da Qwen3-8B çalıştı
  (generate modu, thinking kapalı, greedy, en fazla 256 token). 2.792 isteğin hepsi biçime uydu;
  üretim 64 eşzamanlı istekle ~3 dakika sürdü. Sorgu metni isteğin iki satırı (`server: …\ntool: …`),
  medyan 99 karakter.
- **İlgililik.** Makale isabeti sunucu adı + tool adı eşleşmesiyle sayıyor. Bu yüzden aynı adlı
  sunucudaki aynı adlı tool da ilgili (40 sorguda 2–3 ilgili tool var).
- **Metrik.** Makalenin "accuracy"si top-1 isabeti; burada karşılığı Precision@1. Recall@1, birden
  fazla ilgili tool'u olan sorgularda yarım puan verdiği için biraz daha düşük çıkıyor. Makale
  MCP-tools sonucunu tek bir sayı olarak vermiyor (Figure 5'teki ısı haritası; hiyerarşik eşleşme,
  text-embedding-3-large). Bu yüzden sayılar makaleyle karşılaştırılabilir değil; burada ölçülen,
  skorer'ların kendi aralarındaki sıralaması.
- **w/ inst.** Benchmark talimat taşımıyor; w/ inst ayarında sorgulara şu genel talimat ekleniyor:
  "Given an agent's request for a tool, retrieve the MCP tool that fulfills it."

Tüm korpus (2.792 tool), düz (hiyerarşik değil) retrieval:

| Run | dataset | inst | n | Precision@1 | Recall@5 | NDCG@10 | Recall@10 | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/documentation/concat/nostem | mcp_zero | y | 2792 | 21.60 | 49.22 | 39.15 | 61.22 | 0.093 |
| bm25/documentation/plain/nostem | mcp_zero | n | 2792 | 56.34 | 76.30 | 68.83 | 82.10 | 0.095 |
| dense/emb/qwen3-emb/documentation/plain | mcp_zero | n | 2792 | 45.24 | 69.55 | 60.96 | 78.03 | 7.505 |
| dense/emb/qwen3-emb/documentation/instruct_query | mcp_zero | y | 2792 | 69.73 | 89.28 | 82.01 | 93.61 | 7.645 |
| clm[qwen_60k_skip_neg0]/emb/qwen3-emb/documentation/instruct_query | mcp_zero | y | 2792 | 71.38 | 90.33 | 83.23 | 94.40 | 0.094 |
| clm[qwen_full_skip_neg0_e5]/emb/qwen3-emb/documentation/instruct_query | mcp_zero | y | 2792 | **71.99** | **91.30** | **83.77** | **94.79** | 0.093 |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/example_call/clm | mcp_zero | y | 2792 | 1.47 | 4.48 | 3.63 | 6.45 | 10.083 |
| clm[clm_60k_lr1e-2_negf95]/emb/qwen3-8b/example_call/clm | mcp_zero | y | 2792 | 4.62 | 11.14 | 9.68 | 16.65 | 0.051 |

Satırlar: BM25 (makale ayarı, `--no-stem`) w/ ve w/o inst; Qwen3-Embedding-8B w/o ve w/ inst;
Qwen3-Embedding-8B + skip head (60K ve 206K ToolRet-train çifti; hafta 3'ün checkpoint'leri, MCP
verisi görmedi); CLM sıfır-atış ve fine-tune (60K). p50 sütunu cache durumunu yansıtıyor, gecikme
ölçümü değil.

Top-1 hatalarının dağılımı (aynı sıralamalar üzerinde bir kerelik analiz, w/ inst): Qwen3'te
hataların 17.6 puanı yanlış sunucudaki bir tool, 12.8 puanı doğru sunucudaki başka bir tool;
head'lerle (206K) 15.2 / 12.6; BM25 (w/o inst) 36.0 / 7.6. 100 sorguda hedef tool'un açıklaması
ilgili olmayan başka bir tool'la birebir aynı; bu da top-1 için ~%96'lık bir tavan koyuyor.

**Ablasyon: tool metninde sunucu adı.** Sorgunun `server:` satırı sunucuyu adıyla söylüyor, ama tool
metninde (`documentation`: ad, açıklama, `inputSchema`) sunucu yok. Aynı korpusa her tool'un JSON'una
`"server": <sunucu adı>` eklendi (`data/mcp_zero_server`, sorgular aynı):

| Run | dataset | inst | n | Precision@1 | Recall@5 | NDCG@10 | Recall@10 | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/documentation/plain/nostem | mcp_zero_server | n | 2792 | **80.44** | 93.11 | 88.25 | 95.40 | 0.129 |
| dense/emb/qwen3-emb/documentation/instruct_query | mcp_zero_server | y | 2792 | 78.19 | 92.31 | 87.21 | 95.63 | 0.078 |
| clm[qwen_60k_skip_neg0]/emb/qwen3-emb/documentation/instruct_query | mcp_zero_server | y | 2792 | 79.62 | 93.48 | 88.25 | 96.08 | 0.096 |
| clm[qwen_full_skip_neg0_e5]/emb/qwen3-emb/documentation/instruct_query | mcp_zero_server | y | 2792 | 79.87 | **94.20** | **88.53** | **96.12** | 0.093 |

Sunucu adı Qwen3'e top-1'de +8.5, head'lere +7.9, BM25'e +24.1 puan kazandırıyor. Bu ayarda
BM25 top-1'de önde (80.4 / 79.9); head'ler Recall@5 ve sonrasında önde. Sorgular hedef tool'un
açıklamasından üretildiği için sözcük örtüşmesi yüksek; bu set BM25'e elverişli, "agent kendi
isteğini yazar" kurgusunu ölçüyor. Kullanıcı görevinden gelen sorgular için ToolRet ve
LiveMCPBench daha temsilî.

## LiveMCPBench

Hafta 4'ün ilk koşuları (525 tool, 94 görev, gold = görevde kullanılan tool'lar), buna w/o inst
satırları ve aynı sunucu adı ablasyonu eklendi:

| Run | dataset | inst | n | Recall@5 | NDCG@10 | Comprehensiveness@10 | Recall@5 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/documentation/concat/nostem | livemcpbench | y | 94 | 20.33 | 21.75 | 15.96 | 18.50 | 0.153 |
| bm25/documentation/plain/nostem | livemcpbench | n | 94 | 28.95 | 29.29 | 19.15 | 26.72 | 0.151 |
| dense/emb/qwen3-emb/documentation/plain | livemcpbench | n | 94 | 50.00 | 50.82 | 34.04 | 49.85 | 40.181 |
| dense/emb/qwen3-emb/documentation/instruct_query | livemcpbench | y | 94 | 49.04 | 51.42 | 38.30 | 49.64 | 161.522 |
| clm[qwen_60k_skip_neg0]/emb/qwen3-emb/documentation/instruct_query | livemcpbench | y | 94 | 52.25 | 54.65 | 38.30 | 51.84 | 0.73 |
| clm[qwen_full_skip_neg0_e5]/emb/qwen3-emb/documentation/instruct_query | livemcpbench | y | 94 | 52.05 | 52.84 | 36.17 | 52.01 | 0.183 |
| clm[CLM_v0.1-8B]/emb/qwen3-8b/example_call/clm | livemcpbench | y | 94 | 4.56 | 4.96 | 3.19 | 5.95 | 40.587 |
| clm[clm_60k_lr1e-2_negf95]/emb/qwen3-8b/example_call/clm | livemcpbench | y | 94 | 8.57 | 6.79 | 5.32 | 11.47 | 0.207 |

Tool metninde sunucu adı (`data/livemcpbench_server`):

| Run | dataset | inst | n | Recall@5 | NDCG@10 | Comprehensiveness@10 | Recall@5 cat-macro | p50 ms |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25/documentation/plain/nostem | livemcpbench_server | n | 94 | 31.68 | 32.08 | 22.34 | 29.44 | 0.139 |
| dense/emb/qwen3-emb/documentation/instruct_query | livemcpbench_server | y | 94 | 50.82 | 53.74 | 37.23 | 50.99 | 0.177 |
| clm[qwen_60k_skip_neg0]/emb/qwen3-emb/documentation/instruct_query | livemcpbench_server | y | 94 | **53.72** | **55.26** | **40.43** | **53.06** | 0.102 |
| clm[qwen_full_skip_neg0_e5]/emb/qwen3-emb/documentation/instruct_query | livemcpbench_server | y | 94 | 53.03 | 54.14 | 37.23 | 52.97 | 0.101 |

Kullanıcı görevleri sunucuyu nadiren adıyla söylüyor; kazanç küçük (Recall@5 +1.0 ile +2.7 arası),
ama dört skorer'da da aynı yönde. 94 sorguda bir sorgu ~1 puan; tek başına anlamlı değil,
MCP-Zero'daki etkiyle birlikte okunmalı.

## Komutlar

```bash
# GB10, ~/toolrank (kod: 4ebe966)
docker compose -f deploy/spark/compose.yaml --profile gen up -d qwen3-8b-chat
~/.local/bin/uv run toolrank data pull mcp-zero --gen-workers 64
docker compose -f deploy/spark/compose.yaml stop qwen3-8b-chat

E="$HOME/.local/bin/uv run toolrank eval"
M="--data data/mcp_zero --ks 1,5,10,20"
L="--data data/livemcpbench"
Q="--emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation"
C="--emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b --truncate 2048 --tool-format example_call"
H60=data/heads/qwen_60k_skip_neg0.pt; HALL=data/heads/qwen_full_skip_neg0_e5.pt; CFT=data/heads/clm_60k_lr1e-2_negf95.pt

$E $M --scorer bm25 --no-stem --tool-format documentation --with-inst --out results/mcpzero_bm25.json
$E $M --scorer bm25 --no-stem --tool-format documentation --out results/mcpzero_bm25_noinst.json
$E $M --scorer dense $Q --query-format instruct_query --with-inst --out results/mcpzero_qwen3emb.json
$E $M --scorer dense $Q --out results/mcpzero_qwen3emb_noinst.json
$E $M --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $H60 --out results/mcpzero_qwen3emb_heads.json
$E $M --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $HALL --out results/mcpzero_qwen3emb_headsfull.json
$E $M --scorer clm $C --with-inst --out results/mcpzero_clm.json
$E $M --scorer clm $C --with-inst --clm-ckpt $CFT --out results/mcpzero_clm_ft.json

$E $L --scorer bm25 --no-stem --tool-format documentation --with-inst --out results/livemcp_bm25.json
$E $L --scorer bm25 --no-stem --tool-format documentation --out results/livemcp_bm25_noinst.json
$E $L --scorer dense $Q --query-format instruct_query --with-inst --out results/livemcp_qwen3emb.json
$E $L --scorer dense $Q --out results/livemcp_qwen3emb_noinst.json
$E $L --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $H60 --out results/livemcp_qwen3emb_heads.json
$E $L --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $HALL --out results/livemcp_qwen3emb_headsfull.json
$E $L --scorer clm $C --with-inst --out results/livemcp_clm.json
$E $L --scorer clm $C --with-inst --clm-ckpt $CFT --out results/livemcp_clm_ft.json

# ablasyon: aynı veri, her tool'un JSON'unda önce "server": <Tool.category>; sorgular kopyalanır
for d in mcp_zero livemcpbench; do ~/.local/bin/uv run python -c "
import json, shutil, sys; s, t = sys.argv[1], sys.argv[1] + '_server'
import pathlib; pathlib.Path('data/' + t).mkdir(exist_ok=True)
with open(f'data/{s}/tools.jsonl') as f, open(f'data/{t}/tools.jsonl', 'w') as g:
    for line in f:
        r = json.loads(line); doc = {'server': r['category'], **r['doc']}
        r['doc'], r['documentation'] = doc, json.dumps(doc, ensure_ascii=False); g.write(json.dumps(r, ensure_ascii=False) + '\n')
shutil.copy(f'data/{s}/queries.jsonl', f'data/{t}/queries.jsonl')" $d; done
for d in mcp_zero livemcpbench; do
  S="--data data/${d}_server"; [ $d = mcp_zero ] && S="$S --ks 1,5,10,20"; o=${d/livemcpbench/livemcp}; o=${o/mcp_zero/mcpzero}
  $E $S --scorer bm25 --no-stem --tool-format documentation --out results/${o}_srv_bm25_noinst.json
  $E $S --scorer dense $Q --query-format instruct_query --with-inst --out results/${o}_srv_qwen3emb.json
  $E $S --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $H60 --out results/${o}_srv_qwen3emb_heads.json
  $E $S --scorer clm $Q --query-format instruct_query --with-inst --clm-ckpt $HALL --out results/${o}_srv_qwen3emb_headsfull.json
done

# Mac
toolrank compare results/mcpzero_{bm25,bm25_noinst,qwen3emb_noinst,qwen3emb,qwen3emb_heads,qwen3emb_headsfull,clm,clm_ft}.json \
  --metrics Precision@1,Recall@5,NDCG@10,Recall@10 --cat-macro ''
toolrank compare results/livemcp_*.json --metrics Recall@5,NDCG@10,Comprehensiveness@10 --cat-macro Recall@5
```

## Ortam

GB10; vLLM 0.13.0 (NGC `nvcr.io/nvidia/vllm:26.01-py3`). Retrieval: 8090 Qwen3-8B pooling (bf16,
`--max-model-len 2048`), 8091 Qwen3-Embedding-8B (bf16, 8192). Sorgu üretimi: 8093 Qwen3-8B chat
(compose `gen` profili; bf16, `--max-model-len 4096`, prefix caching, `--gpu-memory-utilization
0.2`, 57K token KV cache), sıcaklık 0, `enable_thinking: false`; üretimden sonra durduruldu. MCP-Zero
tool'ları soğuk cache'le kodlandı (2.792 tool: Qwen3-Embedding 40 s, Qwen3-8B 20 s); LiveMCPBench
koşuları sıcak cache'le. Ham veri sha256'sı ve prompt commit'i `data/mcp_zero/SOURCE.md`'de.

## Sapmalar ve açıklamalar

- **Sorgu üreticisi.** Makale kapalı modeller kullanıyor; burada Qwen3-8B. İstekler hedef
  açıklamanın kısa bir yeniden yazımı, yani görev kolay ve sözcük düzeyinde. CLM'in backbone'u da
  Qwen3-8B, ama bundan bir avantaj görünmüyor (top-1 1.5).
- **Düz ve hiyerarşik eşleşme.** Makale önce sunucuyu (açıklama + özet), sonra o sunucunun
  tool'larını eşliyor. Buradaki tüm skorer'lar tek bir sorgu metniyle bütün tool'ları sıralıyor.
  Sunucu adı ablasyonu bu farkın büyük kısmını kapatıyor.
- **Genel talimat BM25'i düşürüyor.** MCP-Zero'da top-1 56.3'ten 21.6'ya, LiveMCPBench'te Recall@5
  29.0'dan 20.3'e iniyor. Talimattaki sözcükler (tool, agent, request, retrieve, MCP) MCP tool
  metinlerinde sık geçiyor, sorgular da kısa. ToolRet'in göreve özgü talimatlarında durum tersiydi
  (22 → 36). Kapı raporundaki LiveMCPBench BM25 satırı (w/ inst) bu yüzden BM25 için karamsar; w/o
  inst satırı eklendi. Qwen3-Embedding için talimat MCP-Zero'da çok yardımcı (45.2 → 69.7),
  LiveMCPBench'te etkisiz (50.0 / 49.0).
- **Recall@1 ve accuracy.** Makalenin isabet testi Precision@1'e denk. Tabloda Recall@1 yok; 40 çok
  ilgili sorgu yüzünden 0.7 puandan az daha düşük çıkıyor (ör. Qwen3 69.20 ve 69.73).

## Sonraki hafta

- Kapı kararı (senin): `faz0-gate.md` taslağı bu raporla güncellendi. Karar verilince Faz 1
  backlog'u güncellenir; "sunucu adıyla indeksleme" Faz 1'in MCP proxy / ingestion işine girer.
- Açık, opsiyonel: Qwen3-Embedding-8B LoRA (hafta 4), LLM'li `example_call` (hafta 2; `ChatModel`
  portu artık var), replay ve ablasyonlar (hafta 3).
