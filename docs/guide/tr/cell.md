# 1 · Hücreyi kur

**Hedef:** kayıt ile policy'yi çalıştırma arasında hiçbir şeyin kıpırdamadığı bir sahne. Policy pikselleri öğrenir;
tepsi, bir kasa ya da bir kamera bir santim kayarsa, öğrendiği pikseller artık aynı anlama gelmez.

## Masada ne var

| Bileşen | Neden |
| --- | --- |
| Taban plakası (Gridfinity karoları) | Kolu, tepsiyi, kasaları ve direği tek bir ızgaraya kilitler |
| Tepsi, 2 × 3 cep, mat siyah | Siyah üstünde açık renk parçalar: hem kamera hem sen için kolay |
| OK kasası (gri, kolun sağında) ve REJECT kasası (turuncu, kolun solunda) | Farklı renkler, iki hedef asla birbirine benzemesin diye |
| Üst kameralı direk, 66 cm yukarıda | Tepsiyi, iki kasayı ve kolu tek karede görür |
| Gripper'daki bilek kamerası | Çenelerin arasındaki parçayı görür: kavramaları hassas yapan bu |
| Taban plakasında dört AprilTag | Muayenenin cepleri görüntüde milimetre cinsinden bulmasını sağlar |
| 60 parça: jant, dişli, balata, braket × beyaz/turuncu × sağlam / eksik delik / kırık | Öğrenmek için çeşitlilik, muayene için kusurlar |

Her şey tek bir dosyadan üretiliyor, [`hardware/params.yaml`](../../../hardware/params.yaml): baskı dosyaları ve
simülasyon aynı sayıları kullanıyor. Baskı ve montaj: [`hardware/README.md`](../../../hardware/README.md).

## Yazıcın yok mu?

Altı yeri işaretli, koyu renk herhangi bir tepsi ve farklı renkte iki kutu kullan, her şeyi bantla ya da vidayla
sabitle. Kaybettiğin şey `flywheel-inspect` olur (tag'lere ve cep konumlarına ihtiyacı var), yolun geri kalanı değil.

## Kontrol

- Her bileşeni parmağınla it: hiçbir şey kaymıyor.
- Kablolar kamera görüntüsüne düşemiyor, kola takılamıyor.
- Işık yukarıdan ya da önden geliyor, hücrenin arkasındaki bir pencereden değil; tepside kolun sert gölgesi yok.
