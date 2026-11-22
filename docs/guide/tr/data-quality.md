# 5 · Veri kalitesi

Taklit öğrenmesi verideki her şeyi kopyalar: tereddütlerini, kestirmelerini ve sahnedeki her kazayı. Başarısız olan
bir policy neredeyse her zaman sana dataset hakkında bir şey gösteriyordur. Bu kurallar, çiğnendiklerinde bir modeli
ne sıklıkla mahvettiklerine göre sıralı.

## 1. *Nasıl*'da tutarlı, *ne*'de çeşitli ol

| Aynı tut | Bilerek değiştir |
| --- | --- |
| Kolun başlangıç ve bitiş pozu | Cepte hangi parçanın olduğu |
| Her parça tipini nasıl kavradığın (jantı kenarından, balatayı kısa kenarından enine…) | Parçanın cepteki dönüşü |
| Yol: yukarı, karşıya, aşağı, bırak, geri | Diğer ceplerden hangilerinin dolu olduğu |
| Hız: akıcı, episode başına yaklaşık 10 sn | Işık (birkaç bloğu günün başka bir saatinde kaydet) |
| Kamera konumları, kasa konumları, tepsi konumu | Kasaların ne kadar dolu olduğu |

Görevi yapmanın tek bir yolu, defalarca gösterilirse 10 episode'dan öğrenilebilir. İki ayrı yol, her biri 5 kez
gösterilirse öğrenilemez.

## 2. Hedefin tek ipucu cümle olmalı

Policy, verinin izin verdiği her kestirmeyi kullanır. Beyaz parçalar hep sağa gittiyse "beyaz → sağ" diye öğrenir ve
kelimeleri yok sayar. O yüzden her blokta komut verilen kasaya sağlam *ve* kusurlu, beyaz *ve* turuncu parçalar koy.
Böylece cebi ve kasayı tahmin ettiren tek şey cümle olur.

Aynı sebeple her cebi iki kasayla da kaydet ve her cümle için aynı sayıda episode olsun. `scripts/status.sh` dengeyi
gösteriyor.

## 3. Ölü zaman yok, el yok

- Kayıt başlar başlamaz hareket et; başlangıç pozuna döner dönmez → tuşuna bas. Hiçbir şeyin olmadığı kareler
  policy'ye beklemeyi öğretir.
- Kayıt sırasında ellerin, leader kol ve sen iki kamera görüntüsünün de dışında olmalısınız.
- Görüntüde başka hiçbir şey değişmesin: telefon yok, bardak yok, episode'ların yarısında orada duran bir kablo yok.

## 4. (Başta) sadece temiz başarıları tut

Temiz bir episode: tek kavrama, tepsiye çarpma yok, parça kasaya düşüyor, kol geri dönüyor. Gerisi: **←** ve yeniden
yap. Toparlanma davranışı (kayınca yeniden kavramak) değerli, ama onu sonra ekle, bilerek, eğitilmiş policy'nin
gerçekten yaptığı hatalardan yola çıkarak ([8. adım](rollout.md)).

## 5. Kameralardan yapabiliyor olmalısın

Policy iki 640 × 480 görüntü görüyor, başka hiçbir şey değil. Kavrama anında parça iki görüntüde de gripper'ın
arkasında kalıyorsa öğrenecek bir şey yok. Bunu ilk episode'dan önce [kurulumda](setup.md) kontrol et.

## 6. Kayıt ile çalıştırma arasında sahneye dokunma

Aynı kamera açısı, aynı kasa konumları, aynı kalibrasyon dosyaları. Bir kameraya çarptıysan, daha fazla kaydetmeden
önce dataset'ten bir kareyi referans alıp onu yeniden hizala.

## Ne kadar

| Cümle başına episode | Beklenti |
| --- | --- |
| 5 | Doğru bölgeye gider, kavramayı sık sık kaçırır |
| 10 (bu plan, toplam 120) | Çalışan bir başlangıç noktası; LeRobot'un konum başına önerdiği alt sınır |
| 20–30 | Belirgin şekilde daha güvenilir; her yere değil, değerlendirmenin hata gösterdiği yere ekle |

Daha fazla veri sadece policy'nin zayıf olduğu yerde işe yarar. 120 ile eğit, değerlendir, sonra başarısız olan
cümleler için daha fazla kaydet.

## Her bloktan önce

- [ ] Kol başlangıç pozunda, gripper açık
- [ ] Eller ve leader iki kamera görüntüsünün de dışında
- [ ] Doğru cepte bir parça var; hangi kasa olduğunu biliyorsun
- [ ] Kasalar taşmıyor
- [ ] Masada yeni bir şey yok
