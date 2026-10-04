# Faz N — Hafta M raporu (tarih)

## Sonuç (tek cümle)

Bu hafta ne ölçüldü, sayı ne çıktı, karar ne.

## Ölçüler, birimler ve yön

Her tablo başlığında ok yönü (↑ yüksek iyi, ↓ düşük iyi) ve birim: `NDCG@10 ↑ %`, `p50 ↓ ms`,
`token / sorgu ↓`, `ücret ↓ $`. Rapora özgü bir ölçü varsa burada tek satırla tanımla (ne ölçüyor,
birimi, hangi yön iyi). Sıralama ölçüleri yüzde, "fark" sütunları yüzde puanı.

## Ön kayıt (koşudan önce doldurulur)

Karar kuralı koşudan ÖNCE yazılır: hangi set, hangi ölçü, hangi eşik ve her olası sonuçta ne
yapılacağı. İki koşu aynı sorgularda karşılaşıyorsa `toolrank compare --paired A B` çıktısı tabloya
girer ve p değeri kararın parçası olur; "1 puan fark" tek başına karar verdirmez (LiveMCPBench'in
94 sorgusunda top-1 farkının anlamlı olması için 8,5–12,8 puan gerekir).

## Tablo

`toolrank compare results/<...>.json` çıktısı buraya, başlıklarında ok ve birimle; her satırın komutu
aşağıda.

## Komutlar

```bash
# aynen kopyalanabilir, tekrar üretilebilir
```

## Ortam

vLLM sürümü, model, `--max-model-len`, dtype, GPU, cache durumu (soğuk/sıcak), commit hash.

## Sapmalar ve açıklamalar

Beklenen değerden 1 puandan fazla sapma varsa: fark, olası neden (tokenizasyon, stemming, kırpma), denenen şey.

## Sonraki hafta

Plan dosyasındaki bir sonraki kutular; değişen bir şey varsa neden.
