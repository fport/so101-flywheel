# 2 · Kolu hazırla

**Hedef:** follower leader'ı akıcı şekilde taklit ediyor ve iki kamera da doğru görüntüyü veriyor.

## Kurulum

```bash
git clone https://github.com/fport/so101-flywheel && cd so101-flywheel/python
uv sync --extra robot          # LeRobot 0.6.1 with the Feetech servo driver
hf auth login                  # token with write access
```

## Script'lere düzeneğini tanıt

```bash
uv run lerobot-find-port               # unplug one arm when asked; note both ports
uv run lerobot-find-cameras opencv     # saves one frame per camera, note the two indices
```

Dört değeri ve Hugging Face kullanıcı adını `scripts/env.local.sh` dosyasına yaz (sana ait, commit edilmez; şablonu
[`scripts/env.sh`](../../../scripts/env.sh) dosyasının başında). Her script onları oradan okur ve çalıştırmadan
önce tam `lerobot-*` komutunu ekrana basar, yani neyin çalıştığını her zaman görürsün.

## Kalibrasyon (bir kez)

```bash
scripts/calibrate.sh follower
scripts/calibrate.sh leader
```

Kolu her eklemin ortasına getir, Enter'a bas, sonra her eklemi bir uçtan öbür uca gezdir. Kalibrasyon ham servo
tick'lerini iki kolda da aynı açıya eşler; `hardware/calibration/` içine kaydedilir. Kayıt yaptıktan sonra yeniden
kalibre edersen dataset'indeki sayıların anlamı değişir: kayıttan önce yap, arada değil.

## Teleop

```bash
scripts/teleop.sh           # leader drives follower
scripts/cams.sh             # in a second terminal: both cameras at http://localhost:8642
```

Başlamadan önce leader'ı follower'ın pozuna yakın tut: follower bağlanır bağlanmaz leader'ın pozuna sıçrar.

## Kontrol

- Üst kamera: tepsi, iki kasa ve kolun bütün erişim alanı kadrajda, içinde başka hiçbir şey hareket etmiyor.
- Bilek kamerası: iki çene ucu ve aralarındaki boşluk görünüyor.
- Görevi bir kez sadece iki kamera penceresine bakarak yap, masaya bakmadan. Görüntülerden yapamıyorsan policy de
  yapamaz. Yapabilene kadar kameraların yerini değiştir.
- Tek bir al-bırak, takılmadan yaklaşık 10 saniye sürene kadar pratik yap.

**Neden Rerun değil?** `scripts/teleop.sh --cams` LeRobot'un Rerun görüntüleyicisini açar ama Mac'te kontrol döngüsünü
kilitledi ("Sender has been blocked… waiting for space in channel"). Tarayıcı önizlemesi tek küçük bir süreç ve kolu
bloklayamaz. Kayıttan önce kapat: kayıt kameraları kendisi kullanır.