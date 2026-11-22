# Imitation learning nasıl çalışır

Adım sayfaları sana ne yapacağını söylüyor. Bu sayfa neden yaptığını anlatıyor. Bütün proje üç fikrin üstünde duruyor:
behaviour cloning (davranış klonlama), compounding error (biriken hata) ve action chunk.

## Behaviour cloning bir supervised learning

Dataset'in uzun bir tablo. Her satır saniyenin otuzda biri:

```
observation o_t                                   action a_t
  top image, wrist image, 6 joint angles, sentence  →  6 target joint angles (where the leader was)
```

Policy bir fonksiyon: `π(o) → a`. Eğitim, ağırlıkları her satırda `π(o_t)` değeri `a_t`'ye yakın çıkacak şekilde
ayarlar. Ödül yok, "parça kasaya ulaştı" diye bir kavram da yok. Modele hep tek bir şey soruluyor: *bu durumda insan
sonra ne yaptı?* Düşük bir eğitim loss'unun çalışan bir robot anlamına gelmemesi bu yüzden. Gösterdiğin şeyi,
tereddütlerin dahil, aynen geri alırsın.

## Küçük hatalar neden çığ gibi büyür

Sıradan supervised learning'de %2'lik bir hata %2 kalır. Robotta kalmaz, çünkü sonra ne göreceğine policy'nin kendi
çıktısı karar verir:

```
demonstrations:   s0 → s1 → s2 → s3 → …            (states the human visited)
policy:           s0 → s1'→ s2''→ s3'''→ …         (each small error moves it further off the data)
```

Kol gösterilen yoldan 1 cm saptı mı, artık dataset'te olmayan bir durumdadır. Orada tahmini daha kötüdür, bu da onu
daha da uzaklaştırır. Buna *covariate shift* denir. Ross ve Bagnell (2010) bunun bedelini gösterdi: adım başına ε hata
oranıyla T adımda, düz behaviour cloning T · ε değil, **T² · ε**'a kadar kaybedebilir.

Buna karşı üç şey işe yarar ve üçü de bu projede var:

| Çare | Nasıl | Nerede |
| --- | --- | --- |
| Daha fazla durumu kapsa | Parçayı, dönüşü ve tepsi doluluğunu değiştir; "biraz sapmış" hâlâ verinin içinde kalsın | [Veri kalitesi](data-quality.md) |
| Daha az karar | 50 action'ı bir kerede tahmin et: 300 adımlık bir episode 6 ila 15 karara iner | Action chunk, aşağıda |
| Geri dönüş yolunu göster | Policy hata yapsın, kontrolü devral, düzeltmeyi kaydet | DAgger, aşağıda |

## Tek durum, birden çok doğru cevap

Bir dişliyi soldan da sağdan da kavrayabilirsin. İkisi de doğru. İkisiyle birden düz bir kare hata (squared-error)
loss'uyla eğitilen bir ağ *ortalamayı* tahmin eder: dişlinin ortası, kimsenin göstermediği bir kavrama.

```
demo A: approach from the left   ←●
demo B: approach from the right     ●→
MSE average:                        ●      (closes on nothing)
```

İki savunma var:

1. Belirsizliği **sen** kaldırırsın: her parça tipi için tek kavrama, tek yol, her seferinde. Bu veri kalitesinin 1.
   kuralı ve bundan ucuz bir çözüm yok.
2. **Model** birden çok cevapla baş edecek şekilde kurulur. ACT, gösterimler arasındaki farkları emen bir variational
   autoencoder ile eğitilir ama çalışırken tek ve deterministik bir chunk üretir. SmolVLA ve π0 *flow matching*
   kullanıyor: rastgele gürültüyü tutarlı tek bir action dizisine çevirir ve gerçekten iki modun arasına değil,
   birine oturur. Bkz. [SmolVLA'nın içi](learn-vla.md).

## Action chunk

Modern policy'ler kamera karesi başına tek bir action yerine bir *chunk* tahmin eder: sonraki 50 action'ı tek seferde.

- **Kaymak için daha az fırsat.** ACT makalesi bunu simülasyonda ölçtü (iki görev, scripted ve insan verisi): her seferinde tek adım tahmin edince
  %1 başarı, 100'lük chunk'larla %44.
- **Duraklamalar tuzak olmaktan çıkar.** Bir gösterim yarım saniye duraklarsa, tek adımlık bir policy etiketi
  "kıpırdama" olan aynı gözlemleri görür ve sonsuza kadar donup kalabilir. Bir chunk'ın içinde duraklamanın bir
  başı ve sonu vardır.
- **Karşılanabilir.** Bir VLA'nın bir chunk üretmesi için yaklaşık on ağ geçişi gerekir. Çağrı başına elli action
  demek, ağır model saniyede birden az çalışırken kolun yine 30 Hz'de hareket etmesi demek.

Bedeli: bir chunk oynarken kol bakmıyor. 30 fps'te SmolVLA'nın 50 action'ı **1,67 sn'lik open-loop hareket** eder.
Parça bu sürede kayarsa kol önce eski planı bitirir. Çözüm real-time chunking
([SmolVLA'nın içi](learn-vla.md) sayfasında).

## DAgger: kendi hatalarından öğrenmek

Temiz gösterimler sadece iyi yolu gösterir. DAgger (Ross, Gordon, Bagnell 2011), *policy*'nin gerçekten vardığı
durumları toplar:

1. Policy'yi çalıştır.
2. Ters gittiğinde, o durumdan doğru action'ı bir insan verir.
3. Bu çiftleri dataset'e ekle ve yeniden eğit. Tekrarla.

Bu, hata sınırını T² · ε'dan kabaca T · ε'a geri indirir. LeRobot insan kontrollü bir sürümünü sunuyor:

```bash
lerobot-rollout --strategy.type=dagger --policy.path=... --teleop.type=so101_leader ...
```

| Tuş | Ne yapar |
| --- | --- |
| Space | Policy'yi duraklat / devam ettir |
| Tab | Leader kolla düzeltmeni başlat / bitir |
| Enter | Yükle |
| Esc | Dur |

Varsayılan olarak sadece düzeltme pencerelerin kaydedilir, her biri bir episode olarak. Bu, [8. adımdaki](rollout.md)
veri çarkının "düzelt" kolu.

## Masaya ne götürmeli

- Policy en fazla senin kadar tutarlı olabilir.
- Kendi küçük hatalarından sağ çıkmasını sağlayan şey verideki çeşitlilik.
- Robottaki bir hata, dataset'te eksik olan bir durumdur. Git o durumu kaydet.

**Kaynaklar:** Ross & Bagnell, *Efficient Reductions for Imitation Learning* (2010) ·
[DAgger](https://arxiv.org/abs/1011.0686) · [ACT](https://arxiv.org/abs/2304.13705) ·
*Vision Language Models* (Noyan, Farré, Marafioti), böl. 11, s. 500–505 ·
[LeRobot human-in-the-loop docs](https://huggingface.co/docs/lerobot/hil_data_collection)
