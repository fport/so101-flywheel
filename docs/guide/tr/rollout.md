# 8 · Çalıştır ve iyileştir

**Hedef:** policy'nin ne kadar iyi olduğunu gösteren bir sayı ve onu tam da eksik olan veriyle iyileştirmenin bir yolu.

## Çalıştır

```bash
scripts/rollout.sh 3 ok       # the policy has 30 s to do: "put the part from pocket 3 in the OK bin"
```

`scripts/env.local.sh` içindeki policy'yi yükler (`POLICY`, eğitimin yazdığı "Model repo").

Bir elini follower'ın güç kaynağının yakınında tut. İlk çalıştırma genelde kesik kesik ya da yanlış olur; bu da bilgi.
Mac'te model Apple GPU'sunda (`DEVICE=mps`) çalışır ve yavaştır, o yüzden script *real-time chunking* (`--inference.type=rtc`) kullanır:
sıradaki chunk hesaplanırken kol son tahmin edilen chunk'ı uygulamaya devam eder.

## Puanla

Birkaç şanslı denemeye bakıp karar verme. Bakmadan önce testi sabitle:

| | |
| --- | --- |
| Deneme | Cümle başına 2 = 24, kayıtta kullanmadığın parçalarla |
| Sahne | İlkinde tepsi dolu, ikincisinde yarı boş |
| Başarı | Parça komut verilen kasada ve kol geri dönmüş, 30 sn içinde, senin yardımın olmadan |
| Not al | cep, kasa, başarı ve hata tipi: yanlış cep · kaçan kavrama · düşürme · yanlış kasa · takılma |

Başarıları 24 üzerinden, ayrıca cep başına ve kasa başına raporla. 24 deneme sadece "çalışıyor" ile "çalışmıyor"u
ayırır: 18/24 ile 20/24 aynı sonuçtur. Üzerinde iş yapacağın şey hata tipleri:

| Hata | Genelde anlamı |
| --- | --- |
| Yanlış cebe gidiyor | Cümle kullanılmıyor: cümle başına dengeyi kontrol et, karıştırılan çift için daha fazla kaydet |
| Doğru cep, kavramayı kaçırıyor | Verideki kavramalar tutarsız ya da bilek kamerası parçayı göremiyor |
| Yolun ortasında donuyor | Hiç görmediği bir durum: tepsi doluluğunda daha fazla çeşitlilik lazım, ya da gösterimlerde ölü zaman var |
| Öğlen çalışıyor, gece çalışmıyor | Işık: başarısız olduğu ışıkta birkaç blok kaydet |

## Tüm döngü

```bash
uv run flywheel-inspect     # prints one sentence per pocket
scripts/rollout.sh 5 ok                  # run them one by one
```

## İyileştir: veri çarkı (flywheel)

*Tam olarak başarısız olanı* daha fazla kaydet, yeniden eğit, aynı testle yeniden puanla.

- Başarısız cümleler için **daha fazla gösterim**: `scripts/record.sh 4 reject 10`.
- **Düzeltmeler (DAgger):** policy'yi çalıştır, ters gittiği anda leader ile kontrolü devral ve düzeltmeni kaydet.
  Bu, policy'ye temiz gösterimlerden alamayacağı toparlanma verisini verir:
  `lerobot-rollout --strategy.type=dagger ...` ([LeRobot dokümanları](https://huggingface.co/docs/lerobot/inference)).

Her kayıt → eğitim → puanlama → düzeltme turu bir devir. Her devrin puanını bir tabloda tut; o eğri projenin ta
kendisi.
