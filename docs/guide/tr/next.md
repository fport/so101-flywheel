# Baseline'dan sonra: deney planı

Elinde bir dataset, bir SmolVLA checkpoint'i ve bir puan var. Bu sayfa sırada ne yapacağını anlatıyor; sıra, en az
parayla en çok şey öğreten sıra. Her deney **tek bir şeyi** değiştirir ve **aynı sabit testle** değerlendirilir
([bir policy'yi ölçmek](learn-eval.md)).

## Hücre dört katmandan oluşuyor

Bu rehberdeki her model bunlardan birine ait. Hangisine ait olduğunu bilmek deneyleri birbirinden ayrı tutar.

```
1. GÖR        üst kamera karesi ─► her parça nerede, nesi bozuk
                 bugün: AprilTag'ler + ölçüm             dene: LocateAnything
2. KARAR VER  iş emri + görülenler ─► cep başına OK ya da REJECT
                 bugün: flywheel.qc içinde decide()      (kurallar; öğrenilecek bir şey yok)
3. SÖYLE      cep başına bir cümle: "put the part from pocket 3 in the REJECT bin"
4. YAP        cümle + iki kamera + eklemler ─► saniyede 30 action
                 bugün: SmolVLA                          dene: Cosmos 3 Edge
   ─────────────────────────────────────────────────────────────────────────────
   VERİ       4. katman neyden öğreniyor
                 bugün: teleop episode'ları              dene: DAgger düzeltmeleri, Sim-Transfer, RL
```

Neden her şey için tek bir model değil? Çözünürlük yüzünden. Policy her kamerayı 64 token olarak görüyor; 7 mm'lik bir
delik bunun çok altında kalıyor. Muayene ise milimetre başına 4 piksellik, düzleştirilmiş bir kırpıntıyı okuyor. "Nesi
bozuk" sorusunu policy'nin dışında tutmak, küçük bir VLA'nın bu işi yapabilmesini sağlayan şey: sadece bir cebi ve bir
kasayı bulması gerekiyor.

## Merdiven

| | Deney | Neyi değiştiriyor | Maliyet | Ne gerekiyor |
| --- | --- | --- | --- | --- |
| 0 | SmolVLA baseline | | ~4 saat A100 | İlk önce yapılır |
| 1 | Inference ayarları | Sadece chunk'ın uygulanışı | Ücretsiz | Baseline checkpoint'i |
| 2 | Öğrenme eğrisi | Veri miktarı | 2 eğitim daha | Dataset |
| 3 | DAgger turu | 20 ila 30 düzeltme episode'u | Bir öğleden sonra + 1 eğitim | Kol |
| 4 | Grounding | 1. katman | Bir saatliğine bir GPU, kol yok | Kaydedilmiş kareler |
| 5 | Cosmos 3 Edge | 4. katman | 48 GB+ bir GPU, rollout için de GPU'lu bir makine | LeRobot 0.7 |
| 6 | Sim-Transfer verisi | Veri | H100 sınıfı bir GPU, bir haftalık kurulum işi | Simülatör |
| 7 | Pekiştirmeli öğrenme (RL) | 4. katmanın nasıl iyileştiği | Kolda günler | Bir ödül sinyali |

Bunları yukarıdan aşağıya yap. İlk üçü neredeyse bedava ve pahalı olanların değip değmeyeceğini söylüyor.

### 1. Inference ayarları (ücretsiz)

Aynı checkpoint, aynı test, üç çalıştırma:

| Çalıştırma | Ayar | Neye bakıyor |
| --- | --- | --- |
| a | `--inference.type=sync` | Chunk sınırlarında duraksamalar |
| b | `--inference.type=rtc` (`scripts/rollout.sh` bunu kullanıyor) | Akıcı hareket, geç tepkiler |
| c | `rtc`, chunk başına daha az action uygulanıyor | Kayan bir parçaya daha hızlı tepki, daha fazla hesaplama |

Chunk'lı bir policy'nin nasıl davrandığına dair en ucuz ders bu. Arka plan: [SmolVLA'nın içi](learn-vla.md).

### 2. Öğrenme eğrisi

Aynı modeli verinin yarısıyla eğit, sonra tamamıyla eğitilmiş olanla karşılaştır:

```bash
# the first 5 episodes of each 10-episode block: 5 per sentence instead of 10
HALF=$(python3 -c "print(str([b*10+i for b in range(12) for i in range(5)]).replace(' ',''))")
scripts/train.sh --dataset.episodes="$HALF" --job_name=so101-qc-smolvla-half
```

Liste, kayıt sırasına göre tam 10'ar episode'luk 12 blok olduğunu varsayıyor. Episode sildiysen listeyi
`scripts/status.sh` çıktısından kur.

| Sonuç | Anlamı | Sıradaki |
| --- | --- | --- |
| 60 episode, 120 ile aşağı yukarı aynı puanı alıyor | Sınır veri miktarı değil | Başarısız olanı düzelt: DAgger ya da farklı bir model |
| 120 açıkça 60'tan iyi | Hâlâ eğrinin yükselen kısmındasın | Her şeyden önce daha fazla kaydet |

Bu tek sonuç, önce 3. ve 6. deneylerin mi (daha fazla veri) yoksa 5. deneyin mi (daha büyük bir model) geleceğine
karar veriyor.

### 3. Bir DAgger turu

Policy'yi çalıştır, başarısız olduğunda kontrolü devral, sadece düzeltmeleri kaydet, eski + yeni veriyle yeniden eğit,
yeniden test et. En sık görülen hata tipine yönelik yirmi iyi düzeltme, puanı genelde altmış yeni temiz episode'dan
daha çok oynatır. Nasıl: [çalıştır ve iyileştir](rollout.md); neden:
[imitation learning nasıl çalışır](learn-imitation.md).

### 4. Muayenede grounding

Elle yazılmış ölçümleri bir grounding modeliyle değiştir ya da onunla bir kez daha kontrol et; kaydedilmiş karelerde,
robot olmadan. Kontrolden çok algı ilgini çekiyorsa buradan başla. Tam sayfa: [grounding](learn-grounding.md).

### 5. Policy olarak bir world model

Cosmos 3 Edge'i aynı episode'larla LoRA kullanarak fine-tune et ve aynı testi çalıştır. Soru dar: video da tahmin eden
10 kat büyük bir model cep numarasına daha iyi uyuyor mu? Tam sayfa:
[policy olarak world model](learn-world-models.md).

### 6. Yeniden boyanmış sim episode'ları

MuJoCo hücresinde episode üret, Cosmos 3 Sim-Transfer ile yeniden boya, gerçek veriyle karıştır. Sadece 2. deney daha
fazla verinin işe yaradığını söylediyse yapmaya değer. Tam sayfa: [sim-to-real](learn-sim2real.md).

### 7. Pekiştirmeli öğrenme (RL)

Taklit, gösterimlerinin kalitesinde durur. RL, kolun kendi denemelerinden ve bir başarı sinyalinden öğrenerek onları
aşmasını sağlar. LeRobot HIL-SERL ile geliyor (gerçek kolda human-in-the-loop RL). Ödülü üst kamera verebilir:
"parça komut verilen kasada mı?". Buna ancak taklit bir platoya ulaştığında ve kolu çalışır halde bırakabildiğinde başla.

## İki policy'yi 24 denemeyle karşılaştırmak

Yirmi dört deneme *bozuk* ile *çalışıyor*'u ayırabilir. %70 ile %80'i ayıramaz. O yüzden:

- **Denemeleri eşleştir.** İki policy de aynı 24 denemeyi alır: aynı parça, cep, kasa, tepsi doluluğu. İkisinden
  sadece birinin başarılı olduğu denemeleri say. Bunların 7'sini A, 1'ini B kazandıysa bu gerçek bir fark. 4'e 3 ise
  değil.
- **Sadece toplamları değil, hata tiplerini karşılaştır.** "Yanlış cep 6'dan 1'e düştü", toplam neredeyse hiç
  kıpırdamasa bile bir bulgudur.
- **Tek bir şeyi değiştir.** Yeni bir model *ve* yeni veri *ve* yeni ayarlar, tek cevabı olan üç deneydir.

## Her birinde yanılmanın bedeli

| | Başarısız olursa kaybın | Yine de öğrendiğin |
| --- | --- | --- |
| 1, 2, 3 | Bir öğleden sonra | Chunk'lama, veri miktarı ve düzeltmeler kendi kolunda nasıl davranıyor |
| 4 | Bir saat GPU | Genel bir model senin kusurlarını görmeyi nerede bırakıyor |
| 5 | Bir gün ve onlarca dolar | Bir world model nasıl fine-tune edilir ve servis edilir |
| 6 | Bir hafta | LeRobot dataset'leri yazan bir simülatör; RL'in buna zaten ihtiyacı var |

## Daha yeni bir LeRobot

Bu repo LeRobot 0.6.1'e sabitlenmiş: kol onunla kalibre edildi ve test edildi. Cosmos 3 ise 0.7 ya da sonrasını
istiyor. Dataset formatı (v3.0) aynı, yani kaydetmeye 0.6.1'de devam et ve Cosmos deneyini ayrı bir ortamda eğit.
