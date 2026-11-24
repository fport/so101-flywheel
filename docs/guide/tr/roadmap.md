# Rota: on faz

[Ana sayfadaki](index.md) sekiz adım tek bir policy'yi tek bir kola taşıyor. Bu sayfa ise projenin tamamı: o adımlar
nerede duruyor, onlardan sonra ne geliyor ve her fazda alandaki güncel işlerden hangisini deneme fırsatı buluyorsun.

Her fazın bir **çıkış testi** var. Elindeki faz testi geçmeden sıradakine başlama: her biri, bir sonrakinin ihtiyaç
duyduğu sayıyı ya da aracı üretiyor.

**Durum, burada hiçbir şey olduğundan büyük gösterilmesin diye.** 0'dan 3'e kadar olan fazlar bu repoda kuruldu ve
belgelendi; simülatör, muayene ve script'ler çalıştırıldı. 4'ten 9'a kadar olan fazlar, güncel LeRobot dokümantasyonundan
ve model kartlarından derlenmiş bir plan. Bunların hiçbiri henüz bu kolda çalıştırılmadı.

## Harita

| | Faz | Elinde ne oluyor | Süre | Maliyet |
| --- | --- | --- | --- | --- |
| 0 | [Simüle et](#phase-0-simulate) | Herhangi bir donanımdan önce, MuJoCo'da çözülmüş görev | Bir akşam | Ücretsiz |
| 1 | [Kur ve gör](#phase-1-build-and-see) | Basılmış bir hücre, kalibre edilmiş bir kol, üst kameranın okuyabildiği bir tepsi | Bir hafta baskı, bir gün iş | Filament |
| 2 | [Topla](#phase-2-collect) | Hub'da 120 temiz episode | Bir öğleden sonra | Ücretsiz |
| 3 | [Baseline](#phase-3-baseline) | Bir SmolVLA checkpoint'i ve 24 üzerinden bir puan | Bir gün | ~10 $ GPU |
| 4 | [Baseline'ı sık](#phase-4-squeeze-the-baseline) | Sınırın veri mi model mi olduğunu bilmek | İki gün | ~20 $ |
| 5 | [Başka policy'ler](#phase-5-other-policies) | Aynı test, iki ya da üç güncel modelde | Bir hafta | 20 ila 100 $ |
| 6 | [Daha iyi gözler](#phase-6-better-eyes) | Muayenede grounding, otomatik bir hakem | Üç gün, kol yok | Birkaç GPU saati |
| 7 | [World model'ler](#phase-7-world-models) | Video da tahmin eden bir policy | Bir hafta | 48 GB+ bir GPU |
| 8 | [İkiz olarak simülatör](#phase-8-the-simulator-as-a-twin) | Sim'de değerlendirilen policy'ler, eğitimde sim verisi | İki hafta | Yeniden boyama için H100 sınıfı bir GPU |
| 9 | [Kendi denemelerinden öğrenmek](#phase-9-learning-from-its-own-attempts) | Tek bir alt beceride RL, ağırlıklı taklit | İki hafta | Kolun yanında bir CUDA GPU |

Kapanış fazı bir model değil: [tam döngü](#the-full-loop), kimse cümle yazmadan baştan sona ayrılmış bir tepsi.

## Faz 0: simüle et {#phase-0-simulate}

**Hedef.** Herhangi bir şey basmadan önce görevi ve yerleşimi anla.

- **Yap.** [Önce simüle et](simulate.md): sahneyi kur, script'li uzmanı izle, 120 episode'luk planı çalıştır.
- **Öğren.** Episode nedir, bu görev için "başarı" ne demek, tepsi ve kasalar neden oldukları yerde.
- **Çıkış testi.** `flywheel-sim batch` iki satır için de 60/60 yazdırıyor ve
  "put the part from pocket 3 in the REJECT bin" için kolun ne yapması gerektiğini sesli olarak söyleyebiliyorsun.

## Faz 1: kur ve gör {#phase-1-build-and-see}

**Hedef.** Her gün birebir aynı olan bir hücre ve tepsiyi okuyan bir kamera.

- **Yap.** [Hücre](cell.md), [kol ve kameralar](setup.md), [tepsiyi muayene et](inspect.md).
- **Öğren.** Kalibrasyon, kamera yerleşimi, bir homografi pikselleri nasıl milimetreye çevirir.
- **Çıkış testi.** Teleop leader'ı takılmadan takip ediyor, iki kamera da rehberin göstermesi gerektiğini söylediği
  şeyi gösteriyor ve `flywheel-inspect` kendi dizdiğin bir tepsinin altı cebinin hepsi için doğru kasayı söylüyor.

## Faz 2: topla {#phase-2-collect}

**Hedef.** Başkası kaydetmiş olsaydı güveneceğin bir dataset.

- **Yap.** 12 bloğu [kaydet](record.md), [veri kalitesine](data-quality.md) uy, [gözden geçir ve yükle](review.md).
- **Öğren.** [Imitation learning nasıl çalışır](learn-imitation.md), [dataset'in içi](learn-dataset.md).
- **Çıkış testi.** `scripts/status.sh` 10'ar episode'luk 12 blok gösteriyor, her bloktan en az iki episode izledin
  ve dataset başka bir makinede Hub'dan yükleniyor.

## Faz 3: baseline {#phase-3-baseline}

**Hedef.** Düzgün ölçülmüş, sonraki her şeyin karşılaştırıldığı tek bir sayı.

- **Yap.** SmolVLA'yı [eğit](train.md), sabit 24 denemeyi [çalıştır ve puanla](rollout.md).
- **Öğren.** [Bir VLM'in içi](learn-vlm.md), [SmolVLA'nın içi](learn-vla.md),
  [sıfırdan flow matching](learn-flow.md), [bir policy'yi ölçmek](learn-eval.md).
- **Çıkış testi.** Aralığıyla birlikte bir puan ve hata tiplerinin bir tablosu. Düşük bir puan geçerli bir çıkıştır:
  baseline odur.

## Faz 4: baseline'ı sık {#phase-4-squeeze-the-baseline}

**Hedef.** Puanı neyin sınırladığını neredeyse hiç para harcamadan bul.

- **Yap.** [Deney planının](next.md) 1'den 3'e kadar olan deneyleri: inference ayarları, öğrenme eğrisi, bir DAgger turu.
- **Öğren.** Chunk'ın uygulanışı, veri miktarı ve düzeltmeler, her biri puanı *senin* kolunda nasıl oynatıyor.
- **Çıkış testi.** Şu cümleyi kanıtla tamamlayabiliyorsun: "asıl sınır ___ (çok az veri / yanlış
  veri / model / kontrol döngüsü)".

## Faz 5: başka policy'ler {#phase-5-other-policies}

**Hedef.** Sonucun ne kadarının SmolVLA, ne kadarının veri olduğunu gör.

Güncel LeRobot yirmiden fazla policy'yi belgeliyor. İlk önce zamanına değenler, bu kolu daha önce görmüş olanlar,
çünkü onun için bir prior ile başlıyorlar.

| Policy | Neden bu | Başlangıç checkpoint'i |
| --- | --- | --- |
| ACT | Taban: buradaki en küçük policy, dil yok. Tek bir cümleyle eğit | Sıfırdan |
| GR00T N1.7 | NVIDIA'nın 3 B'lik VLA'sı; dokümanlar örnek olarak SO-101'i kullanıyor | `nvidia/GR00T-N1.7-3B` |
| G0.5 | Harekete geçmeden önce bir alt görev yazıyor; bir SO-101 checkpoint'i var | `lerobot/g05_so101` |
| MolmoAct2 | Varsayılan olarak LoRA, 200'den az gösterim için tavsiyeler | `lerobot/MolmoAct2-SO100_101-LeRobot` |
| NAT VLA | 450 M, autoregressive token'lar, belgelenmiş bir 24 GB tarifi | `ahad-j/nat-vla-450m` |

- **Yap.** GPU'lu makinede güncel bir LeRobot ortamı kur (bu repo kayıt için 0.6.1'de kalıyor), *remote inference*'ı öğren (model bir GPU'da, kol
  masanda), sonra tablodakilerden bir ya da ikisini aynı dataset ile eğit.
- **Öğren.** [2026'da alan](learn-landscape.md): aileler ve aralarında neyin farklı olduğu.
- **Çıkış testi.** Aynı 24 denemede eşleştirilmiş bir karşılaştırma, sadece tek bir policy'nin başarılı olduğu denemelerin sayısıyla birlikte.

## Faz 6: daha iyi gözler {#phase-6-better-eyes}

**Hedef.** Algıyı daha az elle yazılmış hale getir ve denemeleri gözle değerlendirmeyi bırak.

- **Yap.** Kaydedilmiş karelerde üç grounding deneyi ([grounding](learn-grounding.md)). Sonra otomatik bir
  hakem: üst kamera kasaların nerede olduğunu zaten biliyor, yani "parça komut verilen kasada mı" bir ölçümdür.
  Bunu, kaydedilmiş deneme videolarında zero-shot bir ödül modeliyle (TOPReward, ROBOMETER) karşılaştır.
- **Öğren.** Genel bir model senin kusurlarını görmeyi nerede bırakıyor; ödül modelleri ilerlemeyi nasıl puanlıyor.
- **Çıkış testi.** En az 48 denemede, kendi kararların ile her otomatik hakem arasındaki uyumun bir tablosu.

## Faz 7: world model'ler {#phase-7-world-models}

**Hedef.** Geleceği video olarak tahmin eden bir policy çalıştır ve ne hayal ettiğine bak.

- **Yap.** FLUX 3 Action ile başla; bir SO-101 checkpoint'i ile geliyor ve tam olarak bizim veri şeklimizi istiyor (30 Hz, altı
  komutlu eklem, bir sahne kamerası ve bir bilek kamerası). Sonra LoRA ile Cosmos 3 Edge
  ([policy olarak world model](learn-world-models.md)).
- **Öğren.** Video ve action'ın birlikte denoise edilmesi, tahmin edilen video bir hata hakkında sana ne söylüyor.
- **Çıkış testi.** Aynı 24 deneme, artı tahmin edilen videoya bakarak açıklanmış üç hata.

## Faz 8: ikiz olarak simülatör {#phase-8-the-simulator-as-a-twin}

**Hedef.** Simülatörü bir demo değil, eğitim ve değerlendirme döngüsünün parçası yap.

- **Yap.**
  1. MuJoCo hücresini bir EnvHub ortamı olarak paketle: `make_env()` fonksiyonu olan tek bir `env.py`. LeRobot o zaman
     herhangi bir checkpoint'i onun içinde değerlendirebilir (`lerobot-eval --env.type=envhub`) ve onu bir robot gibi
     sürebilir (`--robot.type=sim`), gerçek leader kol ile de.
  2. Farkı ölç: aynı policy, sim'de ve gerçek kolda puanlanmış.
  3. Görünüm farkını yeniden boyanmış episode'larla kapat ([sim-to-real](learn-sim2real.md)).
- **Öğren.** Sim ile gerçek arasında ne aktarılıyor, ne aktarılmıyor.
- **Çıkış testi.** Gerçek + üretilmiş veriyle eğitilmiş, gerçek 24 denemede faz 3 baseline'ına karşı puanlanmış bir policy.

## Faz 9: kendi denemelerinden öğrenmek {#phase-9-learning-from-its-own-attempts}

**Hedef.** Tek bir küçük beceride kendi gösterimlerinin kalitesini aş.

- **Yap.** Tek bir kısa alt beceride HIL-SERL, örneğin "3. cepteki parçayı kavra ve kaldır"; başarı dedektörü olarak
  üst kamera. Ayrı olarak, dataset'ini bir ödül modeliyle puanla ve ağırlıklı taklit ile yeniden eğit
  (RA-BC). İkisi de [gerçek kolda reinforcement learning](learn-rl.md) sayfasında.
- **Öğren.** Ödüller, off-policy öğrenme, müdahaleler neden önemli, RL neden kısa bir görev istiyor.
- **Çıkış testi.** Müdahale oranı bir oturum boyunca düşüyor ve öğrenilen beceri o alt beceride en az
  imitation policy'si kadar güvenilir.

## Tam döngü {#the-full-loop}

Yukarıdaki her şey bir parça. Parçalar arada insan olmadan sırayla çalıştığında proje bitmiş olur:

```
tepsi masada
   → üst kamera karesi → muayene (ölçüm, ikinci görüş olarak grounding)
   → decide(): her cep için OK ya da REJECT
   → altı cümle, cep başına bir tane
   → policy her birini uygular
   → otomatik hakem: parça doğru kasada mı?
   → hatalar kaydedilir → düzeltmeler → sonraki eğitim turu
```

O son ok, repoya adını veren çark (flywheel): hücre bir sonraki dataset'ini kendi üretiyor.

## Sadece bir ayın varsa

0'dan 4'e kadar olan fazlar kursun kendisi. Hiçten, ölçülmüş ve iyileştirilmiş bir policy'ye kadar bütün yolu kapsıyorlar
ve yaklaşık otuz dolara mal oluyorlar. Sonra neyde iyi olmak istediğine göre 5 ila 9'dan **birini** seç:

| Önem verdiğin şey | Git |
| --- | --- |
| Modeller ve eğitim | 5, sonra 7 |
| Algı | 6 |
| Simülasyon | 8 |
| Kontrol ve RL | 9 |
