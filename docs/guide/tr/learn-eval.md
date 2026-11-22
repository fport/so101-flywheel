# Bir policy'yi ölçmek

Bir kez çalışan bir robot demosu neredeyse hiçbir şey kanıtlamaz. Bu sayfa, kendini kandırmanı engelleyecek kadar az
istatistik.

## Loss başarı değildir

Eğitim loss'u, modelin kaydedilmiş durumlarda kaydedilmiş hareketleri ne kadar iyi tekrar ettiğini ölçer. Başarı ise
policy'nin kendi yarattığı durumlara bağlı ([compounding error](learn-imitation.md)). Aynı loss'a sahip iki
checkpoint kolda %30 ve %80 alabilir. Kestirme yok: sonuçları gerçek robotta sayarsın.

## "24'te 18" ne kadar kesin?

Her deneme, yanlılığı bilinmeyen bir yazı tura. Wilson aralığı, gördüğünle uyuşan gerçek başarı oranlarının aralığını
veriyor (%95):

```
centre = (p + z²/2n) / (1 + z²/n)
half   = z · sqrt( p(1−p)/n + z²/4n² ) / (1 + z²/n)        with p = successes/n, z = 1.96
```

| Sonuç | %95 aralık |
| --- | --- |
| 18 / 24 (%75) | %55 – %88 |
| 20 / 24 (%83) | %64 – %93 |
| 6 / 24 (%25) | %12 – %45 |
| 45 / 60 (%75) | %63 – %84 |

24'te 18 ile 24'te 20 neredeyse tamamen üst üste biniyor: bu iddia edebileceğin bir iyileşme değil. 6'ya karşı 18
öyle. %75'i %85'ten ayırmak için birkaç yüz deneme lazım, o yüzden buna hiç girişme: 24 denemeyi *bozuk* ile *çalışıyor*'u
ayırmak için kullan, sırada ne yapacağına da hata tiplerine bakarak karar ver.

## Testi çalıştırmadan önce sabitle

Bunların hepsine önce karar ver ve policy sürümleri arasında aynı tut:

- hangi parçalar (bazılarını kayıttan ayrı tut: sadece gördüğü parçaları halleden bir policy onları ezberlemiştir),
- her deneme için hangi cep ve hangi kasa (cümle başına 2, yazılı),
- tepsinin ne kadar dolu olduğu,
- süre sınırı (30 sn) ve neyin başarı sayıldığı (parça komut verilen kasada, kol geri dönmüş, yardım yok).

Denemeleri policy'nin hareketini gördükten sonra seçersen, farkında olmadan yapabildiklerini seçersin.

## İki soru, iki sayı

Görevde birbirinden ayrılabilen iki beceri var. Onları ayrı puanla:

| Soru | Sayı | Düşükse anlamı |
| --- | --- | --- |
| Komut verilen cebe ve kasaya gitti mi? | *komuta uydu* | Cümle kullanılmıyor: bir veri dengesi sorunu |
| Alıp bırakmayı tamamladı mı? | *başarı* | Bir manipülasyon sorunu: kavrama tutarlılığı, kamera görüşü |

Her seferinde 1. cebi kusursuzca OK kasasına boşaltan bir policy'nin manipülasyon becerisi yüksektir ama komuta
uymada 12'de 1 alır. İkisinden hangisinin başarısız olduğunu bilmek, hangi bloğu yeniden kaydedeceğini söyler.

## Hataları tipine göre kaydet

```
trial  pocket  bin     result   failure
1      3       ok      ✓
2      3       reject  ✗        wrong bin
3      5       ok      ✗        missed grasp
```

24 denemeden sonra hata sütunu senin kayıt planın olur. Kitabın büyük sistemler için anlattığı fikir de bu: her adımı
değil, *son durumu* değerlendir ("parça kasada mı?") ve sonucu eğitime geri besle. π*0.6 bunu büyük ölçekte yapıyor:
her denemeyi beklenenden iyi ya da kötü diye etiketliyor ve policy'yi bu etikete göre koşullandırıyor.

## Yargıcı otomatikleştirmek

Üst kamera tepsiyi okumayı zaten biliyor, kasaları da okuyabilir. "Episode'dan sonra 3. cep boş ve REJECT kasasında
bir parça fazla" gibi bir kontrol, her denemede çalıştırabileceğin bir doğrulayıcı. Kayıt → eğitim → değerlendirme
zincirini sen izlemeden dönen bir döngüye çeviren şey bu. Bu repoya adını veren veri çarkı (flywheel) bu ve henüz
kurulmadı.

**Kaynaklar:** Wilson (1927) score interval · [SmolVLA paper](https://arxiv.org/abs/2506.01844) (değerlendirme protokolü) ·
*Vision Language Models*, böl. 11 s. 474–475 ve 509–512
