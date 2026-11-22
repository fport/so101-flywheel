# 4 · Kayıt

**Hedef:** 120 gösterim: sen leader'ı hareket ettirirsin, follower görevi yapar, LeRobot iki kamera akışını, eklem
açılarını ve senin komutlarını saniyede 30 kez, görev cümlesiyle birlikte saklar.

## Plan: 10'arlık 12 blok

Bir blok = bir cümle, örneğin *"put the part from pocket 3 in the OK bin"* (3. cepteki parçayı OK kasasına koy).
Altı cep × iki kasa = 12 cümle, her biri 10 episode. Blok başına yaklaşık 8 dakika, toplamda iki saatten az.

```bash
scripts/record.sh 1 ok          # 10 episodes, pocket 1 -> OK bin
scripts/status.sh               # what is done, and the next command to run
scripts/record.sh 1 reject      # ...
```

Her blok `outputs/datasets/so101-qc` içindeki aynı dataset'e eklenir ve blok bitince Hugging Face hesabına (private)
push edilir; yani bir çökme sana en fazla bir bloğa mal olur.

## Bir blok, adım adım

1. Bloğun cebine bir parça koy. Kol başlangıç pozunda (gripper açık, taban ile tepsi arasında havada).
2. Komutu çalıştır. Bir ses **"Recording episode N"** der: kayıt başladı.
3. Parçayı al, bloğun kasasına bırak, başlangıç pozuna dön.
4. **→** tuşuna bas. Episode biter; ses **"Reset the environment"** der.
5. Şu an hiçbir şey kaydedilmiyor. Sıradaki parçayı cebe koy, elini tekrar leader'a al, kolu başlangıç pozuna getir.
6. Sonraki episode'u başlatmak için **→** tuşuna bas (ya da 15 sn bekle, kendiliğinden başlar). Önceki episode diske şimdi yazılır.

| Tuş | Ne yapar |
| --- | --- |
| → | Bu aşamayı hemen bitir (episode bitti / reset bitti) |
| ← | Bu episode'u at ve yeniden kaydet |
| Esc | Hemen dur, kaydet ve yükle. O ana kadar kaydedileni tutar; bu yüzden parça taşırken değil, reset sırasında bas |

Kavramayı mı beceremedin, tepsiye mi çarptın, dönmeyi mi unuttun? **←** tuşuna bas; reset sırasında fark etsen bile olur. Bir episode'u asla "çoğunlukla
çalıştı" diye tutma.

## Blok içinde cümle hariç her şeyi değiştir

- **Parça:** her episode'da farklı bir tane. Tipleri, renkleri, sağlam ve kırık olanları karıştır. Blok OK diyorsa
  kırık bir parça da OK kasasına gider: kol cümleye uyar, cümlenin ne olacağına muayene karar verir.
- **Açısı:** her seferinde farklı çevir.
- **Diğer cepler:** 1–4. episode'larda tepsi dolu, 5–7'de iki üç komşu, 8–10'da parça tek başına. Çalışma sırasında
  kol iş yaptıkça tepsi boşalıyor ve policy bunların hepsini görmüş olmalı.

Bunun neden önemli olduğu ve kuralların geri kalanı: [veri kalitesi](data-quality.md).

## Bir şeyler ters giderse

| Belirti | Çözüm |
| --- | --- |
| Kamerada `TimeoutError` | Yanlış index (yakındaki bir iPhone sırayı kaydırır): `uv run lerobot-find-cameras opencv`, `scripts/env.sh`'ı düzelt |
| Servodan "no status packet" | Follower'ın 12 V beslemesi ya da gevşek bir servo kablosu |
| Ok tuşları bir şey yapmıyor | macOS ayarlarında terminale Erişilebilirlik izni ver ya da terminali odakta tut |
| Dataset kaldığı yerden devam etmiyor | Son çalıştırma yazma sırasında öldü. Klasörü sil (`rm -rf outputs/datasets/so101-qc`) ve son push edilen kopyayı yeniden indir: `hf download fport/so101-qc --repo-type dataset --local-dir outputs/datasets/so101-qc` |
