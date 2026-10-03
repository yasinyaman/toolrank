# Backlog — Dalga 1 raporu (3 Ekim 2026–)

Dalga 1, Faz 1 kapısından (11 Kasım) önce benimsenmeyi artırmak için öne alınan backlog maddeleri:
60k LoRA kararı (D1.1), küçük backbone'lar (D1.2), GGUF ile GPU'suz yol (D1.3), LangChain 1.x middleware'i
(D1.4), toolrank skill'i (D1.5) ve 0.3.0 (D1.6). Bu rapor maddeler bittikçe büyüyor.

## Sonuç (tek cümle)

D1.3, D1.4 ve D1.5 bitti; D1.2'nin dizüstü kısmı bitti, MCP-Zero ve dev setleri GB10'u bekliyor, D1.1 de.
Varsayılan backbone'un Q4_K_M GGUF'u, 4 GB'lık bir dizüstü GPU'sunda Ollama'yla vLLM'deki bf16 kadar iyi: ToolRet
NDCG@10 59,50'ye karşı 58,90, LiveMCPBench 55,75'e karşı 55,74. Bu dizüstünde yeni bir arama 0,6–0,7 sn sürüyor;
Qwen3-Embedding 0.6B ve 4B 2–14 kat hızlı ama 6–11 puan geride. `ToolrankToolSelector`, LangChain 1.x'in
`create_agent`'ına takılan bir middleware. 1.862 tool'luk katalogda iki görevin ikisinde de model yalnız 2 tool
gördü, doğru tool aralarındaydı; tur başına tek arama yapıldı ve çağrı o aramaya bağlandı. toolrank skill'iyle
yalnız kabuğu olan bir ajan aynı katalogda doğru tool'u bulup çağırdı.

## Ölçüler, birimler ve yön

`gösterilen tool ↓ adet`: bir model çağrısında modele bağlanan tool sayısı. `arama / tur ↓`: bir kullanıcı
mesajında toolrank'a gidilen arama sayısı. `doğru ↑`: görevin tool'u gösterildi ve çağrısı `ok` bitti.
`süre ↓ s`: ajan kurulumu hariç, bir görevin baştan sona süresi. `NDCG@10 ↑ %`, `Recall@k ↑ %`, `cat-macro ↑ %`:
README tablosundaki gibi, talimatlı (w/ inst). `soğuk ↓ ms`: sunucunun daha önce görmediği bir isteğin aranması,
embedding dahil, istekler tek tek (batch 1); `ılık ↓ ms`: aynı istek embedding cache'inden. `kodlama ↓`:
kataloğun ilk kez gömülmesi. `GGUF ↓ GB`: model dosyasının boyutu.

## D1.2 ve D1.3 — dizüstünde küçük backbone'lar ve GGUF

GB10'a erişilemediği için ikisi de 4 GB'lık bir dizüstü GPU'sunda ölçüldü (RTX 3050 Ti Laptop, 14 GB RAM),
Ollama 0.35.1 ile. Modeller GGUF:

- Qwen3-Embedding 0.6B (f16, Q8_0), 4B (Q4_K_M, Q8_0) ve 8B (Q4_K_M, Q8_0): Qwen'in kendi GGUF depolarından.
- `toolrank-emb-v0.2`: Hub'daki `yasinyaman/toolrank-emb-8b@v0.2`, llama.cpp'nin dönüştürücüsüyle önce f16'ya,
  sonra Q8_0 ve Q4_K_M'ye (imatrix'siz).

Her model Ollama'ya `num_ctx 8192` ile girdi. Ollama vLLM'in `truncate_prompt_tokens`'ını yok sayıyor, uzun bir
girdiyi `num_ctx`'te kesiyor ve vLLM gibi başını tutuyor; yani 8192 token'lık sınır iki yanda aynı. Değerlendirme
README satırlarıyla aynı: dense, `documentation` + `instruct_query`, talimatlı, tam katalog üstünde top-100.
Değerlendirmeler dizüstünde koştu, Ollama'ya kendi Tailscale adresinden gidildi (`scripts/gguf_matrix.sh`).

### LiveMCPBench (94 istek, 525 tool)

| Model | GGUF ↓ GB | NDCG@10 ↑ % | Recall@5 ↑ % | Recall@10 ↑ % | 525 tool'un kodlanması ↓ dk |
| --- | ---: | ---: | ---: | ---: | ---: |
| *vLLM bf16, Qwen3-Embedding-8B* | | 53,74 | 50,82 | 61,09 | |
| *vLLM bf16, `toolrank-emb-v0.2`* | | 55,74 | 52,06 | 63,34 | |
| Qwen3-Embedding-0.6B f16 | 1,2 | 49,12 | 47,03 | 55,47 | 0,9 |
| Qwen3-Embedding-0.6B Q8_0 | 0,6 | 49,21 | 47,39 | 55,39 | 0,5 |
| Qwen3-Embedding-4B Q4_K_M | 2,5 | 48,71 | 45,79 | 57,71 | 2,8 |
| Qwen3-Embedding-4B Q8_0 | 4,3 | 49,39 | 45,63 | 56,86 | 4,4 |
| Qwen3-Embedding-8B Q4_K_M | 4,7 | 54,42 | 49,49 | 62,44 | 5,4 |
| Qwen3-Embedding-8B Q8_0 | 8,1 | 53,62 | 50,70 | 61,00 | 8,7 |
| `toolrank-emb-v0.2` Q4_K_M | 5,0 | 55,75 | 52,22 | 62,59 | 5,4 |
| `toolrank-emb-v0.2` Q8_0 | 8,7 | 55,23 | 52,17 | 63,34 | 8,8 |

### ToolRet (7.961 istek, 44.453 tool)

8B'nin bir ToolRet satırı bu dizüstünde 8,4 saat sürdüğü için üç model koşuldu.

| Model | NDCG@10 ↑ % | cat-macro ↑ % | Recall@10 ↑ % | Recall@20 ↑ % | 44.453 tool'un kodlanması ↓ sa |
| --- | ---: | ---: | ---: | ---: | ---: |
| *vLLM bf16, Qwen3-Embedding-8B* | 51,11 | 46,54 | 62,32 | 69,45 | — |
| *vLLM bf16, `toolrank-emb-v0.2`, GB10* | 58,90 | 54,36 | 69,54 | 75,25 | 0,4 |
| Qwen3-Embedding-0.6B f16 | 48,54 | 43,14 | 59,41 | 66,35 | 1,0 |
| Qwen3-Embedding-4B Q4_K_M | 50,41 | 44,99 | 61,80 | 68,56 | 3,6 |
| `toolrank-emb-v0.2` Q4_K_M | 59,50 | 54,51 | 69,85 | 75,67 | 7,3 |

### Gecikme, istekler tek tek

`scripts/latency.py`: LiveMCPBench'te model başına 50, ToolRet'te 100 istek.

| Model | Set | soğuk p50 ↓ ms | soğuk p95 ↓ ms | ılık p50 ↓ ms |
| --- | --- | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B f16 | LiveMCPBench | 74 | 167 | 6,0 |
| Qwen3-Embedding-0.6B Q8_0 | LiveMCPBench | 46 | 133 | 3,0 |
| Qwen3-Embedding-4B Q4_K_M | LiveMCPBench | 326 | 381 | 6,0 |
| Qwen3-Embedding-4B Q8_0 | LiveMCPBench | 412 | 509 | 3,0 |
| Qwen3-Embedding-8B Q4_K_M | LiveMCPBench | 657 | 719 | 3,0 |
| Qwen3-Embedding-8B Q8_0 | LiveMCPBench | 790 | 994 | 6,0 |
| `toolrank-emb-v0.2` Q4_K_M | LiveMCPBench | 668 | 707 | 6,0 |
| `toolrank-emb-v0.2` Q8_0 | LiveMCPBench | 780 | 948 | 3,0 |
| Qwen3-Embedding-0.6B f16 | ToolRet | 93 | 191 | 6,0 |
| Qwen3-Embedding-4B Q4_K_M | ToolRet | 275 | 388 | 9,0 |
| `toolrank-emb-v0.2` Q4_K_M | ToolRet | 571 | 694 | 13,4 |

### Ne çıktı

- **D1.3: kuantizasyon ölçülen her yerde kalite kaybettirmiyor.** `toolrank-emb-v0.2`'nin Q4_K_M'i LiveMCPBench'te
  bf16'yla aynı (+0,01), ToolRet'te 0,60 üstünde (cat-macro +0,15; code −0,13, customized +1,05, web −0,44).
  Q8_0 LiveMCPBench'te 0,51 geride, 94 isteğin gürültüsü içinde. Temel Qwen3-Embedding-8B'de de aynı: Q4_K_M
  +0,68, Q8_0 −0,12. GGUF yolu Mac ve CPU kurulumu olarak belgelendi (`docs/guides/local.md`); önerilen yapı
  Q4_K_M: daha küçük (5,0'a karşı 8,7 GB), daha hızlı (soğuk p50 668'e karşı 780 ms) ve aynı kalitede.
- **D1.2: küçük modeller varsayılanın yerini tutmuyor.** 0.6B ve 4B, LiveMCPBench'te `toolrank-emb-v0.2`
  Q4_K_M'in 6,4–7,0 puan, ToolRet'te 9,1–11,0 puan gerisinde (cat-macro 9,5–11,4). ToolRet'te temel 8B'ye
  yakınlar (0,7–2,6 puan geride): fark çoğunlukla LoRA'dan geliyor. 4B ToolRet'te 0.6B'nin 1,9 puan önünde,
  LiveMCPBench'te ikisi aynı yerde (48,7–49,4).
- **Gecikme.** Bu dizüstünde yeni bir arama 0.6B ile 46–93 ms (p95 133–191), 4B Q4_K_M ile 275–326 ms (p95
  381–388), 8B Q4_K_M ile 571–668 ms (p95 694–719) sürüyor. Daha önce görülen bir istek her modelde 3–13 ms:
  sorgu embedding cache'inden geliyor, kalan iş CPU'da sıralama. 300 ms'nin altında bir p95 isteyen bir kurulum
  için bu donanımda yalnız 0.6B yetiyor.
- **Kodlama.** ToolRet'in 44.453 tool'u dizüstünde 0.6B ile 1,0 saat, 4B Q4_K_M ile 3,6 saat, 8B Q4_K_M ile 7,3
  saat; GB10'da vLLM bf16 ile 0,4 saatti. Dakikada kabaca 740, 205 ve 102 tool. 8B Q4_K_M yüklendiğinde 7,6 GB
  tutuyor, bunun 2,6 GB'ı GPU'da, gerisi CPU'da.
- **`toolrank-vllm` imajı** dizüstünde x86_64'te de derlendi ve duman testinden geçti (sahte vLLM'le giriş
  noktası, kim neyi hangi kullanıcıyla koşturuyor); şimdiye kadar yalnız arm64'te (GB10) denenmişti.

## D1.4 — LangChain 1.x middleware'i

`toolrank.integrations.langchain.ToolrankToolSelector` (`[langchain]` ekstrası) bir `AgentMiddleware`.
`wrap_model_call` / `awrap_model_call` içinde `request.tools`'u son kullanıcı mesajına göre daraltıyor.
İki kullanım şekli var:

- **Ajanın kendi tool'ları.** Herhangi bir LangChain tool listesi `/v1/rank` ile sıralanıyor ve uyarlanabilir
  K ile kesiliyor. LangChain'in `LLMToolSelectorMiddleware`'iyle aynı yerde duruyor, ama ek bir model çağrısı
  yapmıyor.
- **toolrank kataloğu (`toolbox=`).** Katalog tool'ları sunucunun indeksinde `/v1/search` ile seçiliyor.
  `Toolbox`'ın tool'larıyla yapılan çağrılar bu aramaya `search_id` ile bağlanıyor; `toolrank learn` bu
  günlüğü okuyor. Ajanın katalog dışı tool'ları olduğu gibi gösteriliyor.

Her iki şekilde de şunlar hep gösteriliyor:

- konuşmada zaten çağrılmış tool'lar;
- `tool_choice`'un adını verdiği tool;
- `always_include`;
- sağlayıcı tool'ları (dict).

Seçim şu durumlarda yapılmıyor:

- seçilecek tool sayısı `min_tools`'u (20) geçmiyorsa;
- bir tool sağlayıcının kendi tool aramasına bırakılmışsa (`extras["defer_loading"]`);
- kullanıcı metni yoksa;
- toolrank hata verirse ya da `timeout_s` (5 sn) içinde cevap vermezse.

Süre aşılırsa seçim arka planda bitiyor ve cache'e giriyor; sonraki model çağrısı onu kullanıyor. Bir turdaki
model çağrıları aynı seçimi `ttl_s` boyunca cache'ten alıyor. Seçim işi bir havuz iş parçacığında
`contextvars.copy_context()` ile koşuyor; LangGraph thread'inin kimliği bir context değişkeni, onsuz arama
oturumsuz kalırdı.

**Uçtan uca** (`scripts/frameworks_e2e.py --only langchain`): `data/w3` kataloğu (1.862 tool), Qwen3-Embedding-8B
ve v0.1 head'leri. Model betikli: gösterilen tool'lar arasında görevin tool'u varsa onu çağırıyor, sonra cevap
veriyor. Görevlerden biri senkron, öbürü asenkron koşuldu.

| Yol | Görev | Ajanın tool'u | gösterilen tool ↓ | arama / tur ↓ | model çağrısı | doğru ↑ | süre ↓ s |
| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: |
| katalog (`toolbox=`) | Tokyo'da saat (senkron) | 1.862 | 2 | 1 | 2 | ✓ | 0,83 |
| katalog (`toolbox=`) | `dog` etiketli evcil hayvanlar (asenkron) | 1.862 | 2 | 1 | 2 | ✓ | 0,04 |
| kendi tool'ları (`/v1/rank`) | Tokyo'da saat (senkron) | 120 | 120 (seçim atlandı) | — | 2 | ✓ | 60,08 |
| kendi tool'ları (`/v1/rank`) | `dog` etiketli evcil hayvanlar (asenkron) | 120 | 120 (seçim atlandı) | — | 2 | ✓ | 60,11 |

- **Katalog yolu.** Gösterilen tool'lar `time__get_current_time` + `time__convert_time` ve
  `swagger-petstore__findPets` + `addPet`. Turun ikinci model çağrısı cache'ten geldi. Kullanım günlüğünde iki
  çağrı da `link: search_id` ile LangGraph thread'inin oturumunda (`rest:langgraph:e2e-langchain-toolbox-N`).
  Ajanı 1.862 tool'la kurmak 0,55 sn sürdü.
- **Kendi tool'ları yolu, ilk koşu.** 120 tool'un LangChain'in gönderdiği biçimdeki metni cache'te yoktu ve
  embedding sunucusu (GB10) kapalıydı. Middleware tasarlandığı gibi davrandı: `/v1/rank` 30 sn'de zaman aşımına
  düştü, model her iki çağrıda da 120 tool'u gördü, görev yine doğru bitti.
- **İkinci koşu, dizüstündeki Ollama'yla** (Qwen3-Embedding-0.6B f16, head'siz: `--heads none` → `serve
  --clm-ckpt none`; `w3` önce cache'e gömüldü, 1.862 tool 642 bin token, 329 sn). İki yol da gerçek embedding'le
  çalıştı:

| Yol | Görev | Ajanın tool'u | gösterilen tool ↓ | arama / tur ↓ | doğru ↑ | süre ↓ s |
| --- | --- | ---: | ---: | ---: | :---: | ---: |
| katalog (`toolbox=`) | Tokyo'da saat (senkron) | 1.862 | 2 | 1 | ✓ | 2,61 |
| katalog (`toolbox=`) | `dog` etiketli evcil hayvanlar (asenkron) | 1.862 | 6 | 1 | ✓ | 0,39 |
| kendi tool'ları (`/v1/rank`) | Tokyo'da saat (senkron) | 120 | 2 | — | ✓ | 19,99 |
| kendi tool'ları (`/v1/rank`) | `dog` etiketli evcil hayvanlar (asenkron) | 120 | 4 | — | ✓ | 0,03 |

  - İlk sıralamanın 20 sn'si 120 tool metninin ilk kez gömülmesi; o sırada aynı GPU'da ToolRet de koşuyordu.
    Varsayılan `timeout_s` (5 sn) ile bu ilk model çağrısı bütün tool'larla gider, sonraki çağrı arka planda
    biten seçimi kullanır.
  - Kendi tool'ları yolunda çağrılar aramaya bağlanmıyor (`link: none`): `/v1/rank` günlüğe arama olarak
    yazılmıyor. `toolrank learn` için katalog yolu gerekiyor.
- **e2e betiklerinde hata.** `serve_e2e.py`, `platforms_e2e.py` ve `frameworks_e2e.py`'nin başlattığı
  `serve`'e `--emb-model` verilmiyordu. 0.2.0'dan beri sunucunun varsayılanı `toolrank-emb-v0.2` olduğundan,
  8091'e (`qwen3-emb`) yanlış model adı gidiyordu ve v0.1 head'lerinin cache'i de tutmuyordu. Üç betik artık
  `--emb-model` alıyor (varsayılan `qwen3-emb`).

## D1.5 — toolrank skill'i

`examples/skills/toolrank`, yalnız kabuğu olan ajanlar için (Claude Code skill'leri, bash-only harness'ler).
İçinde kısa bir `SKILL.md` (2.000 karakterin altında) ve yalnız standart kütüphaneyle yazılmış iki betik var:
`scripts/search.py` (`/v1/search`) ve `scripts/call.py` (`/v1/call`). Klasör tek başına kopyalanabiliyor.

- Betikler yönlendirmeyi izlemiyor; bearer token başka bir sunucuya gitmiyor.
- `call.py` çıkış kodunu ayırıyor: 1 tool'un kendi hatası, 2 toolrank'ın reddi ya da erişilemezlik.
- `--search-id` verilmezse sunucu çağrıyı aynı istemcinin son aramasına bağlıyor (User-Agent
  `toolrank-skill/1`).
- `tests/test_skill.py` betikleri ayrı süreç olarak, gerçek bir HTTP sunucusunun arkasındaki toolrank
  uygulamasına karşı koşuyor: arama, çağrı, stdin'den argüman, yazma reddi, 404, 401, yönlendirme,
  erişilemezlik.

**Elle deneme** (Mac, `data/w3`, 1.862 tool; Claude Code betikleri bir ajan gibi çalıştırdı):

- "What time is it in Tokyo right now?" araması `time/get_current_time`'ı ilk sıraya koydu (salt okunur).
- `call.py` ile `{"timezone": "Asia/Tokyo"}` saati döndürdü (çıkış 0); günlükte `link: search_id`.
- Geçersiz saat diliminde tool'un hatası ve sunucunun şema ipucu çıktıya geldi (çıkış 1); `--search-id`
  verilmediği için günlükte `link: client`.

## Komutlar

```bash
# Mac; tool ve sorgu vektörleri data/w3/cache'ten (GB10 kapalıydı)
export GB10=<GB10'un Tailscale adresi>
uv run python scripts/frameworks_e2e.py --only langchain --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e_langchain.json
# the same with the laptop's Ollama (0.6B, no heads; w3 tool texts embedded into data/w3/cache first)
uv run python scripts/frameworks_e2e.py --only langchain --data data/w3 --emb-url http://$LAPTOP:11434/v1 \
  --emb-model qwen3-emb-0.6b-f16 --heads none --out results/frameworks_e2e_langchain_06b.json
uv run pytest tests/test_frameworks.py -k langchain
# D1.5: serve on the w3 catalogue, then the skill's scripts as an agent runs them
uv run toolrank serve --data data/w3 --config skill_trial.json --emb-url http://$GB10:8091/v1 --emb-model qwen3-emb --port 8766
cd examples/skills/toolrank && export TOOLRANK_URL=http://127.0.0.1:8766
python3 scripts/search.py "What time is it in Tokyo right now?"
python3 scripts/call.py time/get_current_time '{"timezone": "Asia/Tokyo"}' --search-id <search_id>
uv run pytest tests/test_skill.py
# D1.2 / D1.3, on the laptop from ~/toolrank (uv in ~/.local/bin, Ollama on the laptop's Tailscale address)
hf download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-f16.gguf Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir ~/models/qwen3-emb-0.6b-gguf
hf download Qwen/Qwen3-Embedding-4B-GGUF Qwen3-Embedding-4B-Q8_0.gguf Qwen3-Embedding-4B-Q4_K_M.gguf --local-dir ~/models/qwen3-emb-4b-gguf
hf download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-Q8_0.gguf Qwen3-Embedding-8B-Q4_K_M.gguf --local-dir ~/models/qwen3-emb-8b-gguf
hf download yasinyaman/toolrank-emb-8b --revision v0.2 --local-dir ~/models/toolrank-emb-8b-v0.2
# the LoRA backbone as GGUF: f16, then Q8_0 and Q4_K_M, with llama.cpp's image (no imatrix)
RUN="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $HOME/models:/models ghcr.io/ggml-org/llama.cpp:full-cuda"
$RUN --convert /models/toolrank-emb-8b-v0.2 --outtype f16 --outfile /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-f16.gguf
$RUN --quantize /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-f16.gguf /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-Q4_K_M.gguf Q4_K_M  # and Q8_0
# one Ollama model per build, each with num_ctx 8192 (names as in docs/guides/local.md)
printf 'FROM %s\nPARAMETER num_ctx 8192\n' ~/models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-Q4_K_M.gguf > Modelfile
ollama create toolrank-emb-v0.2-q4_k_m -f Modelfile
export LAPTOP=<dizüstünün Tailscale adresi>; export EMB_URL=http://$LAPTOP:11434/v1 UV=~/.local/bin/uv
MODELS="qwen3-emb-0.6b-f16 qwen3-emb-0.6b-q8_0 qwen3-emb-4b-q4_k_m qwen3-emb-4b-q8_0" bash scripts/gguf_matrix.sh   # LiveMCPBench
MODELS="qwen3-emb-8b-q4_k_m toolrank-emb-v0.2-q4_k_m qwen3-emb-8b-q8_0 toolrank-emb-v0.2-q8_0" bash scripts/gguf_matrix.sh
SETS=toolret MODELS="qwen3-emb-0.6b-f16 qwen3-emb-4b-q4_k_m toolrank-emb-v0.2-q4_k_m" bash scripts/gguf_matrix.sh
$UV run python scripts/latency.py --data data/livemcpbench_server --emb-url $EMB_URL --emb-model <model> --n 50 \
  --scorer dense --truncate 8192 --tool-format documentation --query-format instruct_query   # ToolRet: --n 100
```

## Ortam

- Mac'te `toolrank serve` (`--emb-model qwen3-emb`, v0.1 head'leri), `data/w3` indeksi ve embedding cache'i sıcak.
- langchain 1.4.3, langchain-core 1.6.6, langgraph 1.2.12.
- Test paketi (D1.5 sonrası): 278 geçti, 1 atlandı; torch'suz 264 geçti, 13 atlandı.
- D1.2 / D1.3: dizüstü NVIDIA RTX 3050 Ti Laptop (4 GB), 14 GB RAM, 20 iş parçacığı, Ubuntu 26.04 x86_64.
  Ollama 0.35.1, her modelde `num_ctx` 8192; dönüştürme `ghcr.io/ggml-org/llama.cpp:full-cuda` ile. Koşular
  3 Ekim 08:28–23:00 arası, sırayla. ToolRet'in ilk satırı koşarken dizüstünde D1.4'ün LangChain denemesi ve
  `toolrank-vllm` imajının derlemesi de çalıştı; kodlama süresini etkileyebilir, kaliteyi değil. Gecikme
  ölçümleri (22:54–23:00) tek başına koştu.

## Sapmalar ve açıklamalar

- **D1.1 ertelendi.** GB10'a 3 Ekim'de ne SSH ne 8091 üzerinden ulaşılabiliyor (zaman aşımı). 60k koşusunun
  durumu bilinmiyor; erişim gelince ilk iş `data/logs/lora_60k.log`. Karar kuralı aynı kalıyor.
- **GPU'lu maddeler.** D1.2 ve D1.3 GB10 yerine 4 GB'lık bir dizüstü GPU'sunda (RTX 3050 Ti Laptop, 14 GB RAM)
  yapıldı. D1.3'ün hedef donanımı zaten bu. ToolRet üç modelle koşuldu; MCP-Zero ve dev setleri dizüstünde yok,
  sorguları GB10'da üretildi. İkisi de GB10'u bekliyor.
- **D1.3'ün karar kuralı harfiyen tamamlanmadı.** Kural "Q8_0 her sette 1 puan içinde" diyor. Q8_0 yalnız
  LiveMCPBench'te ölçüldü (−0,51); ToolRet'te yalnız Q4_K_M koşuldu, MCP-Zero hiç ölçülmedi. Karar yine de
  verildi: Q4_K_M daha kayıplı kuantizasyon ve ölçüldüğü iki sette de bf16'nın gerisinde değil. MCP-Zero teyidi
  GB10'la.
- **Toplu gecikme sayıları yanıltıcı.** `toolrank eval`'in 64'lük partilerle verdiği sorgu başı p50, Ollama'nın
  modeli belleğe yüklediği süreyi de içerebiliyor; 0.6B f16 LiveMCPBench'te toplu 321 ms, tek tek 74 ms. Gecikme
  için yalnız `scripts/latency.py`'nin tek tek ölçümleri kullanıldı.
- **llama.cpp'nin kendi sunucusu.** `llama-server --embeddings` embedding modunda her girdi token'ı için bir çıktı
  satırı tutuyor, Qwen3'te token başına yaklaşık 0,6 MB; 8192 token'lık bir girdi yaklaşık 5 GB GPU belleği
  istiyor ve kesmeyi de yok sayıyor. Sığmayan uzun bir girdi sunucuyu hata döndürmek yerine çökertti. Ölçümler bu
  yüzden Ollama'yla yapıldı; Ollama belleğini planlıyor ve sığmayanı CPU'ya bırakıyor.

## Sonraki adımlar

- D1.1, GB10'a erişim gelince: 60k koşusunun kararı.
- D1.2 ve D1.3'ün GB10 kısmı: MCP-Zero ve dev setleri, GGUF yapılarıyla ve küçük modellerle.
- GGUF'ların Hub'a yüklenmesi (`yasinyaman/toolrank-emb-8b-GGUF`), 0.3.0'la; herkese açık adım, ayrı onayla.
  Kılavuzdaki `TODO(launch)` işareti o güne kadar sürümü durduruyor.
- Apple Silicon'da Ollama ölçülmedi; kılavuz bunu söylüyor.
