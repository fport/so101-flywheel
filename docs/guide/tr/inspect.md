# 3 · Tepsiyi muayene et

**Hedef:** tepsinin tek bir fotoğrafından, altı cebin her biri için bir kasa. Kararı veren kısım bu; kol sadece
uygular. İkisini ayrı tutunca yanlış bir karar ile beceriksiz bir kavrama, ayrı ayrı düzeltilebilen iki ayrı sorun olur.

```bash
uv run flywheel-inspect                       # top camera, now
uv run flywheel-inspect --parts L-T-02 B-W-07 J-W-08 J-W-06 D-W-02 D-W-06   # score it: IDs in pocket order
uv run flywheel-inspect --image outputs/inspect/<time>/frame.png          # a saved frame
```

1. cep kolun solunda, kola en yakın olan; 4. cep onun arkasında. Parça ID'si her parçanın altına kazınmış.

## Nasıl çalışıyor

1. **Tag'leri bul.** Dört AprilTag bilinen milimetre konumlarında duruyor, yani kamera nerede olursa olsun tek bir
   homografi tepsideki milimetreleri görüntü piksellerine eşler.
2. **Altı kırpım çıkar**, her cep için bir tane, hepsi aynı ölçekte (4 px/mm).
3. **Ölç**, siyah tepsi üstündeki açık renk parçanın silüetinde: delikler (içinden tepsi görünür), jant kolları,
   dişli dişleri, dış hattan eksik bir köşe, renk.
4. **Bir VLM'e sor** (`ANTHROPIC_API_KEY` tanımlıysa Claude), saymakla bulunamayanı: parça tipini ve ikinci bir
   görüşü. Anahtar yoksa tip bilinsin diye `--parts` ver.
5. **Karar ver**: renk iş emrindeki renk değilse (`--order W|T`), bir delik eksikse ya da parça kırıksa REJECT.
   Değilse OK.

Sonunda policy'ye verilecek cümleleri yazar, her cep için bir tane.

## Kontrol

`outputs/inspect/<time>/overlay.jpg` dosyasını aç: turuncu kareler ceplerin üstüne oturmalı. Sonra `--parts` ile üç
dört farklı tepside çalıştır; güvenmeden önce geçmen gereken çıta her seferinde `bin 6/6`.

## Sırada

Ölçüm adımı bu dört parça için elle yazılmış. Bir *grounding* modeli ise nesneleri bir metin prompt'undan bulur:
bkz. ["Sırada ne denenir" sayfasındaki LocateAnything](next.md#grounding-locateanything).
