# Grounding: modele nerede diye sormak

Bir sınıflandırıcı *ne* sorusuna cevap verir. Bir robotun ise *nerede*ye ihtiyacı var: hangi cep, hangi piksel, köşesi
kırık olan hangi parça. Bir ifadeyi koordinatlara çevirmeye *grounding* deniyor. Bu sayfa
[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) modelinin bunu nasıl yaptığını ve bu
hücrede elle yazılmış kodun yerini nerede alabileceğini anlatıyor.

Buradaki hiçbir şey henüz bizim karelerimizde çalıştırılmadı. Modelle ilgili bilgiler model kartından.

## Muayene bugün nesneleri nasıl buluyor

`flywheel-inspect` bir şeyin yerini bulmak için hiç öğrenme kullanmıyor:

1. Dört AprilTag tepsinin görüntüdeki konumunu verir (bir homografi).
2. Altı cebin her biri kesilip çıkarılır ve milimetre başına 4 piksel olacak şekilde düzleştirilir.
3. Delikler, jant kolları, dişler ve köşeler silüetten ölçülür.

Kesin ve hızlı, ve sadece programlanmış olanı biliyor: dört parça tipi, iki kusur, görünür kalması gereken tag'ler.
Ona beşinci bir parça gösterirsen söyleyecek bir şeyi olmaz.

## Grounding modeli nedir

YOLO gibi bir dedektörün, eğitim sırasında kararlaştırılmış sabit bir sınıf listesi vardır. Bir grounding modeli ise
sınıfı *çalışma anında metin olarak* alır: "gear", "the orange part", "a part with a broken corner".

LocateAnything bunun için yapılmış bir vision-language model:

| Parça | Ne |
| --- | --- |
| Vision encoder | MoonViT |
| Dil modeli | Qwen2.5-3B-Instruct |
| Bağlayıcı | Bir MLP projector, [bir VLM'in içi](learn-vlm.md) sayfasındaki gibi |
| Boyut | 3 B parametre |
| Eğitim verisi | 12 M görüntü, yaklaşık 140 M sorgu, 785 M kutu; doğal sahneler, robotik, sürüş, GUI'ler ve dokümanlar |

Tek yeni fikri koordinatları nasıl yazdığı. Sıradan bir VLM bir kutuyu token token yazar:
önce `x1`, sonra `y1`, sonra `x2`, sonra `y2`. LocateAnything dördünü de tek bir paralel adımda tahmin eder (*parallel box
decoding*); kart, önceki yaklaşımlara göre 2.5 kata varan throughput'u buna bağlıyor.

## Onunla konuşmak

Prompt'lar sabit cümleler. Koordinatlar görüntüye göreli, 0 ile 1000 arasında tam sayılar olarak geri gelir.

| Ne istiyorsun | Prompt |
| --- | --- |
| Bazı sınıfların bütün örnekleri | `Locate all the instances that matches the following description: gear</c>rim.` |
| Tek bir şey | `Locate a single instance that matches the following description: the orange gear.` |
| Bir ifadeye uyan her şey | `Locate all the instances that match the following description: a part with a broken corner.` |
| Kutu yerine bir nokta | `Point to: the gear in the top-left pocket.` |

```python
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")       # helper class from the model card
result = worker.detect(image, ["gear", "rim", "brake pad", "bracket"])
boxes = LocateAnythingWorker.parse_boxes(result["answer"], width, height)   # 0..1000 → pixels
```

Karttan, sonuçları değiştiren iki ayrıntı:

- Örnek kod **örnekleme yapıyor** (temperature 0.7). Aynı görüntü iki seferde farklı kutular verebilir. Bir ölçüm için
  örneklemeyi kapat.
- Üç mod var: `fast`, `slow` ve `hybrid` (varsayılan). `fast` için basit sahnelere uygun deniyor;
  siyah zemin üstündeki bir tepsi de öyle bir sahne.

## Kutudan cebe

Piksel cinsinden bir kutu henüz bir karar değildir. Zaten hesapladığımız tepsi homografisi herhangi bir pikseli tepsi
üstündeki milimetrelere, oradan da bir cep numarasına çevirir:

```
"Point to: the gear with a missing tooth."  →  (412, 630) in 0..1000      (illustrative numbers)
        → pixel (264, 302) in the 640 × 480 frame
        → tray homography → (31 mm, −42 mm)
        → pocket 2
        → "put the part from pocket 2 in the REJECT bin"   → the policy
```

Akılda tutulacak kalıp bu: grounding modeli kolu asla sürmez. Bir *cep numarası* üretir, onu policy'ye cümle
taşır.

## Üç deney, en ucuzu önce

Üçü de kaydedilmiş kareler üzerinde çalışır, robot gerekmez. `outputs/inspect/*/frame.png` ile her cepteki bilinen
parça birlikte cevap anahtarıdır.

| | Soru | Neyle puanlanır |
| --- | --- | --- |
| **G1. Parçaları bul** | Dolu her cep için tam olarak bir kutu döndürüyor mu, tipi de doğru mu? | Kaydedilmiş bütün karelerde cep başına doğruluk |
| **G2. Kusurları bul** | "a part with a missing hole" doğru cepleri seçiyor mu, başkasını seçmeden? | Parça ID'lerine karşı precision ve recall |
| **G3. Bir tarifi izle** | "the orange rim" doğru cebe düşüyor mu? | Yazılı 30 ifadelik bir listede doğruluk |

Her birini aynı karelerde `flywheel-inspect` ile karşılaştır. Geçilmesi gereken baseline ölçüme dayalı pipeline, ve
güçlü bir baseline: referans tepside altı cebin hepsi için kasayı doğru okuyor.

## Dürüstçe, ne beklemeli

- **G1 çalışmalı.** Mat siyah bir tepsi üstündeki parçalar kolay bir sahne.
- **Zor olan G2.** Kayıt çözünürlüğünde top kamera milimetre başına yaklaşık 1.3 piksel görüyor, yani 7 mm'lik bir
  delik 9 piksel genişliğinde. Kareyi yeniden boyutlandıran bir model onu görmeyebilir. Yapamadığı sonucuna varmadan
  önce ona düzleştirilmiş 4 px/mm'lik cep kırpımlarını ver ya da 1280 × 960'ta çek.
- **"Eksik" zor bir kelime.** Grounding modelleri var olan şeylerin yerini bulur. Orada olmayan bir delik bir
  yokluktur. "a rim with five holes" diye sorup diğerlerini kusurlu saymak daha iyi çalışabilir.
- **Cevap vermemesi gerekirken verebilir.** Kart, hiçbir şey eşleşmediğinde modelin nasıl davrandığını söylemiyor.
  Kusurlu parça içermeyen kareler de ekle ve yanlış kutuları say.

## Doğrulukla ilgili olmayan sınırlar

| | |
| --- | --- |
| Lisans | NVIDIA ticari olmayan lisans: sadece araştırma ve değerlendirme. Bu rehber için uygun, bir ürün için değil |
| Donanım | Linux ve bir NVIDIA GPU, bfloat16. Yayımlanmış tek bellek rakamı, bir A100'de dört 4K görüntülük bir batch için 11.7 GB |
| Herkese açık demo | [Space](https://huggingface.co/spaces/nvidia/LocateAnything) ilk bakış için en hızlı yol. Yüklediğin her şeyi herkese açık say |

## Hücrede nereye oturur

```
top camera frame
   ├─ AprilTags → homography ──────────────┐
   ├─ measurement (today) ─► defect type ──┤
   └─ grounding (experiment) ─► box, point ┤
                                           ▼
                              decide(): OK or REJECT per pocket
                                           ▼
                              one sentence per pocket → policy
```

Grounding ölçümün yanında ikinci bir görüştür, onun yerine geçen bir şey değil. İkisinin anlaşamadığı yer ilginç
veridir: elle bakılacak kareler onlar.

## Masaya ne götürmeli

- Grounding dili koordinatlara çevirir. Homografi koordinatları bir cebe çevirir.
- Önce kaydedilmiş karelerde test et. Hiçbir maliyeti yok ve kol gerektirmiyor.
- İfadeleri ve kareleri ilk çalıştırmadan önce sabitle, tıpkı [policy testinde](learn-eval.md) olduğu gibi.

**Kaynaklar:** [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) model kartı ·
[Eagle](https://github.com/NVlabs/Eagle) · *Vision Language Models* (Noyan, Farré, Marafioti), konum bulma
s. 87–94
