# Bir vision-language modelin içi

VLA, robot bağlanmış bir VLM'dir; o yüzden buradan başla. Bir VLM'in üç parçası ve bir hilesi var.

```
image ──► vision encoder ──► projector ──► ┐
                                           ├──► language model ──► text
text  ──► tokenizer ─────► embeddings ───► ┘
```

## 1. Vision encoder pikselleri token'a çevirir

Bir Vision Transformer görüntüyü kare patch'lere böler ve her patch'i tek bir vektöre, bir *görsel token*'a çevirir.
Sayısı basit bir aritmetik:

| Görüntü | Patch | Token |
| --- | --- | --- |
| 256 × 256 | 16 × 16 | 16 × 16 = 256 |
| 512 × 512 (SmolVLA'nın girdisi) | 16 × 16 | 32 × 32 = 1 024 |

SmolVLM ailesindeki encoder **SigLIP**. Görüntü–açıklama çiftleriyle eğitilmiş, yani vektörleri bir dil modeli için
zaten bir şey ifade ediyor.

## 2. Projector onları kelimeye benzetir

Bir dil modeli sabit genişlikte vektörler bekler (SmolVLM2-500M için 960). Projector, çoğu zaman tek bir lineer
katman, her görsel token'ı bu genişliğe getirir. Ondan sonra görüntü, metinle aynı diziye konmuş bir sıra "görsel
kelime"den ibaret. Bu tasarıma *unified sequence* ya da early fusion deniyor: özel bir düzenek yok, ikisini dil
modelinin sıradan self-attention'ı karıştırıyor.

Eski alternatif (Flamingo) dil modelini dondurulmuş tutar ve görüntüye bakan cross-attention katmanları ekler. Kitap
ikisini karşılaştırıyor ve hangisini seçtiğinden çok verinin ve eğitim ayarlarının önemli olduğu sonucuna varıyor.

## 3. Dil modeli akıl yürütmeyi yapar

Normal bir transformer decoder. Görsel kelimeleri ve metin token'larını okur, sonraki token'ı tahmin eder. Bir VLM'in
dişliler, kasalar ve *left* kelimesi hakkında "bildiği" her şey burada ve encoder'da duruyor. Bir VLA bunların hepsini
bedavaya devralır.

## Hile: görüntü başına daha az token

Attention maliyeti dizi uzunluğunun *karesiyle* büyür. Kamera başına 1 024 token, iki kamera, her kontrol adımında:
bir robot için fazla yavaş. **Pixel shuffle** komşu token'ları daha geniş tek bir token'da birleştirir:

```
32 × 32 tokens × 1 width   →   16 × 16 × 4   →   8 × 8 × 16
       1 024                      256               64 tokens
```

Pixel shuffle tek başına hiçbir şey atmıyor, yeniden diziyor: uzay genişliğe dönüşüyor. Projector sonra her geniş
token'ı dil modelinin 960 sayısına sıkıştırıyor; ince detay orada kayboluyor. SmolVLA kenar başına 4 çarpanı kullanıyor, yani
her kamera karesi **64 token** tutuyor. Bizim iki kameramız 128 token, bir paragraf metinden az.

Doküman modellerindeki *tiling* de aynı ödünleşimden çıkıyor (büyük bir görüntüyü 256 piksellik karolara böl: 15 karo
şimdiden 3 840 token). Robot bunun tersini yapar: düşük çözünürlük, yüksek frekans.

## "Dondurulmuş" ve "fine-tune" burada ne demek

| Parça | Parametre (SmolVLA) | Bizim fine-tune'da |
| --- | --- | --- |
| SigLIP vision encoder | 86 M | dondurulmuş |
| Dil modeli (32 katmanın ilk 16'sı), embedding'ler, projector | ~260 M | dondurulmuş |
| Action expert + küçük projeksiyonlar | ~100 M | **eğitiliyor** |

Dondurulmuş ağırlıklar değişmez, yani model senin kolunu öğrenirken dişlinin neye benzediğini unutamaz. Kitabın daha
hafif alternatif LoRA (donuk katmanlara eklenen küçük, eğitilebilir matrisler) hakkındaki cümlesi: "daha az öğrenir ve
daha az unutur". LoRA'yı bir VLM'i yargıç olarak uyarlamak için ya da daha büyük Cosmos 3 policy'si için kullanırsın.

## Bir VLM'e *ne* değil *nerede* diye sormak

Muayene adımı konum istiyor. Modeller buna farklı formatlarda cevap veriyor:

| Model ailesi | Çıktı |
| --- | --- |
| OWLv2, GroundingDINO | Piksel cinsinden kutular ve bir güven skoru. Eşiği ayarlamak sana kalmış: kitaptaki örnekte 0.3 ile 3 kutu, 0.01 ile 78 kutu çıkıyor |
| PaliGemma | Metnin içinde konum token'ları, `<loc0324><loc0242>…`, koordinatlar 0–1023'e normalize |
| LocateAnything | 0–1000 ölçeğinde kutular ya da noktalar |
| Genel VLM'ler | Metin ya da JSON olarak yazılmış koordinatlar: kullanmadan önce bir şemayla doğrula |

`flywheel-inspect` bu soruyu AprilTag'lerle atlatıyor: cep konumları milimetre cinsinden biliniyor, yani VLM'in sadece
bir kırpıntıda *ne* olduğunu söylemesi yetiyor. Tag'lerin yerine bir grounding modeli koymak
[grounding](learn-grounding.md) sayfasında anlatılıyor.

**Kaynaklar:** *Vision Language Models* (Noyan, Farré, Marafioti): token'lar ve pixel shuffle s. 131–134, mimariler
s. 229–249, konum bulma s. 87–94, LoRA s. 191–199 · LeRobot 0.6.1 kaynak kodu, `policies/smolvla/smolvlm_with_expert.py`
