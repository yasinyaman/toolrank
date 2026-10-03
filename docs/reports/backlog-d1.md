# Backlog — Dalga 1 raporu (3 Ekim 2026–)

Dalga 1, Faz 1 kapısından (11 Kasım) önce benimsenmeyi artırmak için öne alınan backlog maddeleri:
60k LoRA kararı (D1.1), küçük backbone'lar (D1.2), GGUF ile GPU'suz yol (D1.3), LangChain 1.x middleware'i
(D1.4), toolrank skill'i (D1.5) ve 0.3.0 (D1.6). Bu rapor maddeler bittikçe büyüyor.

## Sonuç (tek cümle)

D1.4 bitti: `ToolrankToolSelector`, LangChain 1.x'in `create_agent`'ına takılan bir middleware. 1.862 tool'luk katalogda
iki görevin ikisinde de model yalnız 2 tool gördü, doğru tool aralarındaydı; tur başına tek arama yapıldı ve
çağrı o aramaya bağlandı. D1.1 ertelendi: GB10'a 3 Ekim'de erişilemiyor.

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
- **Kendi tool'ları yolu ölçülemedi.** 120 tool'un LangChain'in gönderdiği biçimdeki metni cache'te yoktu ve
  embedding sunucusu (GB10) kapalıydı. Middleware tasarlandığı gibi davrandı: `/v1/rank` 30 sn'de zaman aşımına
  düştü, model her iki çağrıda da 120 tool'u gördü, görev yine doğru bitti. Bu yol test paketinde süreç içinde
  çalışan bir toolrank'a karşı sınanıyor (`test_langchain_selector_ranks_through_a_served_toolrank`); gerçek
  embedding'le uçtan uca koşu bir embedding sunucusu açılınca yapılacak.
- **e2e betiklerinde hata.** `serve_e2e.py`, `platforms_e2e.py` ve `frameworks_e2e.py`'nin başlattığı
  `serve`'e `--emb-model` verilmiyordu. 0.2.0'dan beri sunucunun varsayılanı `toolrank-emb-v0.2` olduğundan,
  8091'e (`qwen3-emb`) yanlış model adı gidiyordu ve v0.1 head'lerinin cache'i de tutmuyordu. Üç betik artık
  `--emb-model` alıyor (varsayılan `qwen3-emb`).

## Komutlar

```bash
# Mac; tool ve sorgu vektörleri data/w3/cache'ten (GB10 kapalıydı)
export GB10=<GB10'un Tailscale adresi>
uv run python scripts/frameworks_e2e.py --only langchain --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e_langchain.json
uv run pytest tests/test_frameworks.py -k langchain
```

## Ortam

- Mac'te `toolrank serve` (`--emb-model qwen3-emb`, v0.1 head'leri), `data/w3` indeksi ve embedding cache'i sıcak.
- langchain 1.4.3, langchain-core 1.6.6, langgraph 1.2.12.
- Test paketi: 275 geçti, 1 atlandı; torch'suz 261 geçti, 13 atlandı.

## Sapmalar ve açıklamalar

- **D1.1 ertelendi.** GB10'a 3 Ekim'de ne SSH ne 8091 üzerinden ulaşılabiliyor (zaman aşımı). 60k koşusunun
  durumu bilinmiyor; erişim gelince ilk iş `data/logs/lora_60k.log`. Karar kuralı aynı kalıyor.
- **GPU'lu maddeler.** D1.2 ve D1.3 GB10 yerine 4 GB'lık bir dizüstü GPU'sunda (RTX 3050 Ti Laptop, 14 GB RAM)
  yapılacak. D1.3'ün hedef donanımı zaten bu. Bu donanımda 8B ancak GGUF ile ve kısmen CPU'da çalışır, bu
  yüzden ToolRet'in 44k tool'u yerine MCP setleri ve dev setleri ölçülecek.

## Sonraki adımlar

D1.5 (toolrank skill'i, Mac). Ardından D1.2 ve D1.3 dizüstünde. D1.1, GB10'a erişim gelince.
