# Sim-to-real: Cosmos 3 Sim-Transfer

120 episode'u teleop ile kaydetmek bir öğleden sonranı alır. Bir simülatör bir gecede 12 000 tane üretebilir, hem de
kusursuz etiketlerle, üstelik kimsenin tepsiyi yeniden dizmesi gerekmez. İşin püf noktasının bir adı var: *reality gap*.
Bu sayfa onu daraltmanın belirli bir yolunu anlatıyor: simüle edilmiş videoyu senin masana benzeyecek şekilde yeniden
boyamak.

Buradaki hiçbir şey henüz çalıştırılmadı. Modelle ilgili bilgiler
[Cosmos3-Nano-Sim-Transfer kartından](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer); pipeline bizim tasarımımız.

## Açığın üç parçası var

| Açık | Bizim hücrede | Kim kapatabilir |
| --- | --- | --- |
| **Görünüm** | MuJoCo'nun düz renkleri ve sert gölgeleri, karşısında bir webcam'in gürültüsü, parlaması ve bulanıklığı | Randomizasyon ya da üretici bir model: bu sayfa |
| **Dinamik** | Simüle edilmiş bir servo hedefini birebir izler; bir STS3215'in boşluğu vardır, yük altında sarkar ve geriden gelir | Sistem tanılama ya da gerçek veri |
| **Algılama ve zamanlama** | Simüle edilmiş kareler anında gelir ve senkronizedir; USB kameralar ikisi de değildir | Gerçek veri |

Yeniden boyanmış bir video sadece ilk satırı düzeltir. Bütün tasarımı belirleyen de bu: üretilmiş episode'lar gerçek
olanlara bir *ek*tir, asla onların yerine geçmez.

## Elimizde zaten ne var

`flywheel-sim`, basılan parçalarla aynı `hardware/params.yaml` dosyasından hücrenin bir MuJoCo kopyasını kurar: tepsi,
kasalar, direk, gerçek konumlarında iki kamera. Onu script'li bir uzman çözer.

```bash
uv run flywheel-sim batch      # every part to both bins, prints a success table
```

Yani simülatör bize bir insanın veremeyeceğini şimdiden veriyor: action'ları kesin olarak bilinen sınırsız episode ve
her cepte her parça.

## Klasik çözüm: randomize et

Domain randomization her episode'da renkleri, ışıkları, dokuları ve kamera pozunu değiştirir; böylece gerçek dünya
sadece bir varyasyon daha gibi görünür. Ucuzdur ve basit sahnelerde işe yarar, ama görüntüler sentetik olduğunu hemen
belli eder; önceden eğitilmiş bir VLM ise milyonlarca gerçek fotoğraf ve çok az düz gölgelendirilmiş render görmüştür.

## Üretici çözüm: geometriyi koru, pikselleri yeniden boya

Cosmos3-Nano-Sim-Transfer bir *kontrol videosu* ve bir metin prompt'u alır, kontrolü izleyen fotogerçekçi bir video
üretir.

| | |
| --- | --- |
| Kontrol (biri) | Derinlik haritaları, segmentasyon maskeleri, kenar haritaları ya da bulanıklaştırılmış görüntüler |
| Prompt | Sahnenin nasıl görünmesi gerektiğini anlatan bir metin |
| Çıktı | 480p, 30 fps, en fazla 600 kare (20 s) |
| Model | 16 B parametre, Cosmos 3 Nano'nun birkaç adımlık distile edilmiş bir sürümü |
| Hız | Bir H100'de 14.6 kare/s, bir GB200'de 24.5 |
| Çalışma şekli | İstek gönderdiğin bir Docker sunucusu (vLLM-Omni ya da SGLang). Linux |
| Lisans | OpenMDW 1.1, ticari kullanıma izin var |

Asıl mesele bu iş bölümü: *fizikten, durumdan ve etiketlerden simülatör sorumlu kalır; model sadece boyar.* 137. karedeki
action hâlâ simülatörün 137. karede uyguladığı action'dır.

## "Kontrolü izliyor" ne kadar iyi?

NVIDIA, çıktının kendisine verilen kontrole ne kadar yakın olduğunu raporluyor (PAI-Bench, Sim-Transfer ile sade
Cosmos 3 Nano karşılaştırması):

| Kontrol | Sim-Transfer | Temel model | Anlamı |
| --- | --- | --- | --- |
| Segmentasyon mIoU ↑ | 0.74 | 0.64 | Nesneler aşağı yukarı maskenin dediği yerde kalıyor |
| Derinlik hatası (SI-RMSE) ↓ | 0.62 | 0.84 | Derinlik temel modele göre daha iyi izleniyor |
| Kenar F1 ↑ | 0.34 | 0.21 | Temel modelden iyi, ama ince kenarlar sadece gevşek biçimde izleniyor |

Bu son satır bir kalite kontrol hücresi için önemli. Eksik bir 7 mm'lik delik ya da kırık bir köşe kenar düzeyinde
bir detaydır ve yeniden boyamadan sağ çıkmayabilir. *Policy* için bu sorun değil: cümle hangi cep ve hangi kasa
olduğunu zaten söylüyor, yani kolun parçada neyin yanlış olduğunu değil, parçanın nerede olduğunu görmesi gerekiyor.
*Muayene* için ise sorun: `flywheel-inspect`'i üretilmiş karelerle eğitme ya da test etme.

## Bu hücre için pipeline

```
flywheel-sim (scripted expert)
   ├─ actions, joint states ───────────────────────────────┐
   └─ depth or segmentation video, top and wrist           │
            │                                              │
            ▼  Sim-Transfer + prompt                       │
      photoreal top and wrist video                        │
            │                                              ▼
            └──────────────► LeRobot dataset (same keys, same sentences as fport/so101-qc)
                                              │
                    real episodes ───────────►│ train on both
                                              ▼
                                   the same real 24-trial test
```

Dört şeyin yapılması gerekiyor, şu sırayla:

1. **Sim'den derinlik ve segmentasyon.** MuJoCo'nun renderer'ı ikisini de üretir (`enable_depth_rendering`,
   `enable_segmentation_rendering`); `flywheel-sim` bugün sadece RGB kaydediyor.
2. **Bir sim episode yazıcısı**, LeRobot formatında; gerçek dataset'in feature adları, fps'i ve görev cümleleriyle.
3. **Yeniden boyama işi.** İki kamerada 10 s'lik 120 episode için bu 72 000 kare eder; yayımlanan hızla tek bir H100'de
   yaklaşık 80 dakika.
4. **Bir prompt**, *bizim* masamızı anlatan: mat siyah tepsi, gri ve turuncu kasalar, beyaz bir SO-101, ışık. Yazarken
   referans birkaç gerçek karedir.

## Neyin ters gitmesi muhtemel

- **İki kamera birbirini tutmaz.** Kart çoklu görüş tutarlılığı hakkında hiçbir şey söylemiyor. Top ve wrist ayrı ayrı
  yeniden boyanıyor, yani aynı parça her görüşte farklı bir doku alabilir. Denenecek bir şey: iki görüşü tek bir kontrol
  videosunda yan yana döşemek, Cosmos 3 policy'lerinin girdilerini döşediği gibi.
- **Kol bizim kolumuza benzemez.** Bir derinlik haritası "siyah servolu beyaz PLA" demez. Bunu prompt'un demesi gerekir,
  o da episode'dan episode'a kayabilir. Her gövdeye tutarlı bir renk veren segmentasyon kontrolü daha sağlam bir
  seçim.
- **Dinamik açığına dokunulmuyor.** Simüle edilmiş gripper 0.2 s'de, gerçeği 0.5 s'de kapanıyorsa, policy yanlış
  zamanlamayı kusursuz gerçekçilikte resimlerden öğrenir.
- **Başka bir şey için yapılmış.** NVIDIA, Isaac Lab'de kapalı döngü RL'i ve policy değerlendirmesini anlatıyor; model
  orada kareleri canlı boyuyor. Bir imitation dataset'ini çevrimdışı yeniden boyamak bizim kullanımımız, belgelenmiş
  bir kullanım değil.
- **Donanım.** Sadece H100 sınıfı GPU'larda ve RTX PRO 6000'de test edilmiş; bellek belirtilmemiş. 16 B parametreyle
  48 GB'lık ya da 80 GB'lık bir GPU planla.

## Meseleyi çözen deney

Üç policy, aynı mimari, aynı gerçek test:

| Eğitim verisi | Sana ne söyler |
| --- | --- |
| 120 gerçek | Baseline |
| 60 gerçek | İkinci 60 gerçek episode'un ne kadar değerli olduğunu |
| 60 gerçek + 600 üretilmiş | Üretilmiş verinin eksik 60'ın yerini tutup tutamayacağını |

Üçüncüsü ikincisini geçerse sim verisi işe yarıyor. Birincisine de yetişirse teleop'un yarısının yeri
doldurulabiliyormuş. İkincisinden *kötüyse* üretilmiş veri policy'yi gerçeklikten uzağa çekiyor; daha ucuz çözüm
randomize edilmiş sim ya da düpedüz daha fazla gerçek episode.

## Masaya ne götürmeli

- Sim-Transfer karelerin nasıl göründüğünü değiştirir. Simüle edilmiş robotun nasıl hareket ettiğini düzeltemez.
- Her zaman gerçek kolda değerlendir. Yeniden boyanmış bir simülatördeki başarı bir sonuç değildir.
- Üretilmiş episode'ları ayrı bir dataset'te tut. Karışım oranını değiştirmek isteyeceksin.

**Kaynaklar:** [nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) model kartı ·
[cosmos-framework](https://github.com/NVIDIA/cosmos-framework) ·
[Domain Randomization](https://arxiv.org/abs/1703.06907) · [MuJoCo rendering](https://mujoco.readthedocs.io/en/stable/python.html)
