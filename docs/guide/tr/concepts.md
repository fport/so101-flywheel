# Tek sayfada kavramlar

| Terim | Bu projede |
| --- | --- |
| **Teleoperation** | Sen leader kolu hareket ettirirsin, follower onu taklit eder. Gösterimler böyle yapılır |
| **Episode** | Kaydedilmiş tek bir deneme: bir parça, başlangıç pozundan yine başlangıç pozuna |
| **Observation (gözlem)** | Policy'nin her adımda gördüğü: iki görüntü + altı eklem açısı |
| **Action** | Çıktısı: altı hedef eklem açısı (burada: leader'ın olduğu yer) |
| **Policy** | Gözlem + cümle → action fonksiyonu. Eğittiğin şey bu |
| **Imitation learning / behaviour cloning (taklit öğrenmesi)** | Policy'yi, aynı durumda senin yaptığını üretecek şekilde eğitmek. Ödül yok |
| **Distribution shift (dağılım kayması)** | Policy hiç göstermediğin durumlara kayar; sonra hataları üst üste biner. Çeşitlilik ve düzeltmeler bu yüzden önemli |
| **Action chunking** | Tek bir action yerine sonraki 50 action'ı bir kerede tahmin etmek: daha akıcı, küçük hatalar adım adım birikmez |
| **VLM** | Vision-language model: görüntü kodlayıcı + projektör + dil modeli. Görüntü okur, metin yazar |
| **VLA** | Action head'i olan bir VLM: görüntüleri ve bir cümleyi okur, robot action'ları yazar. SmolVLA, π0, GR00T |
| **Flow matching** | SmolVLA'nın action head'i böyle çalışır: gürültüden başla, birkaç adımda bir action chunk'a dönüştür |
| **Fine-tuning** | Önceden eğitilmiş ağırlıklardan başlayıp kendi küçük dataset'inle eğitmek. Burada VLM donuk kalır, action expert eğitilir |
| **Grounding** | Kelimeleri görüntüdeki yerlere bağlamak: "dişli" için kutular ya da noktalar |
| **Rollout** | Policy'yi gerçek robotta çalıştırmak |
| **DAgger** | Policy'yi çalıştır, başarısız olunca düzelt, düzeltmeleri veriye ekle, yeniden eğit |
| **Veri çarkı (flywheel)** | Kayıt → eğitim → değerlendirme → başarısız olanı topla → yeniden eğitim. Her tur bir sonrakini ucuzlatır |
| **World model** | Kameranın sırada ne göreceğini tahmin eden model. Cosmos 3 bunlardan biri; action da üretebiliyor |
| **Sim-to-real** | Simüle veriyle eğit, gerçek robotta çalıştır. Aradaki fark çoğunlukla nesnelerin nasıl göründüğü ve nasıl kaydığı |

## Okuma: *Vision Language Models* (Noyan, Farré, Marafioti)

Bu proje için şu sırayla oku; sayfa numaraları PDF sayfaları.

| Oku | Sayfalar | Ne için |
| --- | --- | --- |
| Böl. 11, VLA'lar | 499–516 | SmolVLA nasıl kurulmuş, action chunk'lar, asenkron kontrol döngüsü |
| Böl. 4, VLA verisi | 146–149, 160, 175–185 | Eğitimden önce kontroller: kare hızı, eklem limitleri, kareler ile eklemler arasındaki zamanlama |
| Böl. 2, konum bulma | 87–94 | Açık sözlüklü tespit, işaret etme (pointing), konum token'ları: muayene adımı için |
| Böl. 11, yapılandırılmış noktalar | 484–487 | Bir VLM'den doğrulanmış koordinat almak |
| Böl. 5, LoRA / QLoRA | 186–204 | Küçük bir VLM'i doğrulayıcı olarak uyarlamak |
| Böl. 3 ve 6 | 131–134, 235–251 | Görüntü token'ları, tiling, pixel shuffle; SmolVLM'in mimarisi |

Şimdilik atla: Böl. 8 (dokümanlar), Böl. 10 (any-to-any), Böl. 11'deki agent kısmı.

Kitaptan bu projenin doğrudan kullandığı üç fikir:

1. Bir VLA, bir VLM artı bir action expert'tir; VLM her chunk için bir kez çalışır, expert denoising yapar (s. 507–511).
2. Bir kare ile ona ait eklem okuması arasındaki 80 ms'lik bir kayma bile yanlış davranış öğretir: eğitimden önce
   zamanlamayı kontrol et (s. 506–507).
3. Başarı kontrolünde her adımı değil, son durumu değerlendir ("parça kasada mı?") (s. 474–475).
