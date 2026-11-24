# 0 · Önce simüle et

**Hedef:** herhangi bir şey basmadan ya da kablolamadan önce görevin tamamının çalıştığını gör.

Repoda hücrenin bir MuJoCo kopyası var; basılan parçalarla aynı `hardware/params.yaml` dosyasından üretiliyor.
İçinde script'li bir uzman görevi çözüyor. Bir laptop'a ihtiyacın var, başka hiçbir şeye değil.

Tarayıcıda etkileşimli bir 3D sürümü de var: [flywheel.getporti.com](https://flywheel.getporti.com). Hücrenin
etrafında dön ve kaydedilmiş bir episode'u oynat.

## Çalıştır

```bash
git clone https://github.com/fport/so101-flywheel && cd so101-flywheel
uv sync
uv run flywheel-sim build                  # params.yaml → sahne + 76 parçalık katalog
uv run flywheel-sim view                   # etkileşimli görüntüleyici: döndürmek için sürükle
uv run flywheel-sim render --out cell.png  # tek kare: üst kamera | bilek kamerası
uv run flywheel-sim episode --part J-T-06  # tek episode: video + json + diziler
uv run flywheel-sim batch                  # 120 episode'luk plan, yaklaşık 80 sn
```

`batch` şöyle bitmeli:

```
                       success  Wilson 95%
correct command         60/60   [0.94, 1.00]
opposite (followed)     60/60   [0.94, 1.00]
  rim                   30/30   [0.89, 1.00]
  gear                  30/30   [0.89, 1.00]
  pad                   30/30   [0.89, 1.00]
  bracket               30/30   [0.89, 1.00]
failure_type: none
```

Çıktılar `outputs/sim/<run>/` içine düşüyor: her episode için parçayı ve sonucu içeren bir `.json`, bir de eklem
durumlarını ve action'ları içeren bir `.npz` (yaklaşık 200 adım × 6 eklem).

## Neye bakıyorsun

| Sahnede | Ne olduğu |
| --- | --- |
| Kol | Resmi SO-101 modeli: STS3215 servo parametreleriyle, konum kontrollü altı eklem |
| Siyah tepsi | Numaralı altı cep, 2 × 3. 1. cep, kolun gördüğü haliyle yakın-solda |
| İki kasa | Gri OK, turuncu REJECT |
| Dört tag | Üst kameranın tepsiyi bulmak için kullandığı AprilTag'ler |
| İki kamera | Direkte `top`, gripper'da `wrist`. Gerçek hücrenin kaydettiği ikisinin aynısı |
| Parçalar | Jantlar, dişliler, balatalar, braketler; beyaz ve turuncu, bazılarında eksik bir delik ya da kırık bir köşe var |

Parça kimlikleri `J-W-03` gibi okunur: bir jant (`J`), beyaz (`W`), 3 numara. Bir parçanın OK mu REJECT mi olduğu
parçanın bir özelliği değil. *İş emrine* bağlı: emir beyaz istiyorsa kusursuz bir turuncu parça rettir.

## Tablonun iki satırı

- **Doğru komut.** Parça, iş emrinin ima ettiği kasaya gider.
- **Ters (uyuldu).** Kola bilerek *yanlış* kasa söylenir ve başarı, itaat etmiş olması demektir.

İkinci satır, dataset'te neden 6 değil 12 cümle olduğunun sebebi. İyi görünen parçaları hep OK'ye gönderen bir policy
parçaları yargılamayı öğrenmiştir. Biz cümleye uyan bir policy istiyoruz, yargılamayı da muayeneye bırakıyoruz.

## Simülatörün bize şimdiden öğrettikleri

Bunlar tek bir parça bile basılmadan önce sim'den çıktı ve her biri gerçek hücreyi değiştirdi.

| Sim'de bulunan | Gerçek hücredeki değişiklik |
| --- | --- |
| Alçak parçalarda cep duvarı çenelerin tabana ulaşmasını engelliyor | Her cebin iki yanında 18 mm'lik bir parmak yuvası |
| Kısa eksen boyunca kapanan çeneler komşu parçaya çarpıyor | Kapanma ekseni tepsinin uzun kenarına paralel; braketler 90° döndürülmüş duruyor |
| Kol yerçekimi altında 5 ila 7 mm sarkıyor | Kavrama noktaları pay bırakıyor; hiçbir şey milimetrik yüksekliğe bağlı değil |
| Tek bir noktaya bırakılan parçalar kasa kenarının üstüne kadar yığılıyor | Bırakma noktası kasanın içine yayılıyor |
| Tepsiyle aynı hizadaki kasalar kenar ceplerde çeneyi engelliyor | Kasalar tepsinin bir karo gerisinde duruyor |
| Kasa duvarı bir tag köşesini üst kameradan saklıyor | Yakındaki iki tag kendi karolarında dışa doğru kaydırıldı |
| Bir parçayı üst kameraya doğru kaldırıp tutmak bu bilekle neredeyse hiç çalışmıyor | Muayene, herhangi bir alıştan önce bütün tepsiyi bir kez, yukarıdan okuyor |

Bu aşamada bir simülatör bunun için var: yanılmanın ucuz olduğu bir yer.

## Sana söyleyemeyecekleri

- **Gerçek servolar nasıl davranıyor.** Simüle edilmiş kol hedefine ulaşıyor; gerçeğinde boşluk ve gecikme var.
- **Her şey nasıl görünüyor.** Düz renkler ve tek bir temiz ışık, karşısında webcam gürültüsü ve parlama.
- **Öğrenilmiş bir policy çalışıyor mu.** Buradaki uzman parçanın tam konumunu simülatörden okuyor. O bir
  script, bir policy değil ve sadece yerleşimin çözülebilir olduğunu kanıtlıyor.

Bu farklar [sim-to-real](learn-sim2real.md) sayfasının konusu; bu simülatörü eğitimin parçası yapmak ise
[rotanın 8. fazı](roadmap.md#phase-8-the-simulator-as-a-twin).

## Ne zaman tamam

- [ ] `flywheel-sim batch` iki satırda da 60/60 yazdırıyor.
- [ ] Görüntüleyicide ya da 3D sitede, iki kameradan da bir episode izledin.
- [ ] "Ters" satırının neden var olduğunu açıklayabiliyorsun.

Sahne, ground-truth etiketler ve modelin sınırları hakkında daha fazla ayrıntı: [`docs/sim.md`](../../sim.md).
