# Backlog — Dalga 1 raporu (3 Ekim 2026–)

Dalga 1, Faz 1 kapısından (11 Kasım) önce benimsenmeyi artırmak için öne alınan backlog maddeleri:
60k LoRA kararı (D1.1), küçük backbone'lar (D1.2), GGUF ile GPU'suz yol (D1.3), LangChain 1.x middleware'i
(D1.4), toolrank skill'i (D1.5) ve 0.3.0 (D1.6). Bu rapor maddeler bittikçe büyüyor.

## Sonuç (tek cümle)

D1.4 ve D1.5 bitti. `ToolrankToolSelector`, LangChain 1.x'in `create_agent`'ına takılan bir middleware. 1.862 tool'luk katalogda
iki görevin ikisinde de model yalnız 2 tool gördü, doğru tool aralarındaydı; tur başına tek arama yapıldı ve
çağrı o aramaya bağlandı. toolrank skill'iyle yalnız kabuğu olan bir ajan aynı katalogda doğru tool'u bulup
çağırdı. D1.1 ertelendi: GB10'a 3 Ekim'de erişilemiyor.

## Ölçüler, birimler ve yön

`gösterilen tool ↓ adet`: bir model çağrısında modele bağlanan tool sayısı. `arama / tur ↓`: bir kullanıcı
mesajında toolrank'a gidilen arama sayısı. `doğru ↑`: görevin tool'u gösterildi ve çağrısı `ok` bitti.
`süre ↓ s`: ajan kurulumu hariç, bir görevin baştan sona süresi.

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
```

## Ortam

- Mac'te `toolrank serve` (`--emb-model qwen3-emb`, v0.1 head'leri), `data/w3` indeksi ve embedding cache'i sıcak.
- langchain 1.4.3, langchain-core 1.6.6, langgraph 1.2.12.
- Test paketi (D1.5 sonrası): 278 geçti, 1 atlandı; torch'suz 264 geçti, 13 atlandı.

## Sapmalar ve açıklamalar

- **D1.1 ertelendi.** GB10'a 3 Ekim'de ne SSH ne 8091 üzerinden ulaşılabiliyor (zaman aşımı). 60k koşusunun
  durumu bilinmiyor; erişim gelince ilk iş `data/logs/lora_60k.log`. Karar kuralı aynı kalıyor.
- **GPU'lu maddeler.** D1.2 ve D1.3 GB10 yerine 4 GB'lık bir dizüstü GPU'sunda (RTX 3050 Ti Laptop, 14 GB RAM)
  yapılacak. D1.3'ün hedef donanımı zaten bu. Bu donanımda 8B ancak GGUF ile ve kısmen CPU'da çalışır, bu
  yüzden ToolRet'in 44k tool'u yerine MCP setleri ve dev setleri ölçülecek.

## Sonraki adımlar

D1.2 ve D1.3 dizüstünde sürüyor. D1.1, GB10'a erişim gelince.
