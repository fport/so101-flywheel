# Gerçek kolda reinforcement learning

Imitation learning seni kopyalar. Senden iyi olamaz ve parçanın kasaya ulaşıp ulaşmadığını hiç öğrenmez.
Reinforcement learning (RL) ise *sonuçtan* öğrenir. Bu sayfa, bunun fiziksel bir kolda binlerce saat harcamadan nasıl
çalıştığını ve bu hücrede neye mal olacağını anlatıyor.

Buradaki hiçbir şey bizim kolumuzda çalıştırılmadı. İş akışı LeRobot'un
[HIL-SERL](https://huggingface.co/docs/lerobot/hilserl) ve reward model dokümanlarından.

## Taklide göre ne değişiyor

| | Taklit | Reinforcement learning |
| --- | --- | --- |
| Neyden öğreniyor | İnsanın ne yaptığından | Sonrasında ne olduğundan |
| Sinyal | Her kare için bir action | Bir reward (ödül), çoğu zaman sadece sonda 1 ya da 0 |
| Veri nereden geliyor | Senden, teleop ile | Policy'nin kendi denemelerinden |
| Tavan | Senin gösterimlerin | Görevin kendisi |
| Ana risk | Gösterilen yoldan sapmak | Keşif: kol bir şeyler dener, kötü olanlar dahil |

Gerçek donanımda RL'in bu kadar uzun süre nadir kalmasının sebebi son satır. Deneyerek öğrenen bir policy'nin çok
denemeye ihtiyacı var ve ilk denemeler çırpınmaya benzer.

## Onu uygulanabilir kılan üç fikir

**1. Off-policy öğrenme.** Algoritma (Soft Actor-Critic, SAC), hangi policy üretmiş olursa olsun, saklanmış her
transition'dan öğrenebilir; off-policy bu demek. Her deneme bir *replay buffer*'a, yani tekrar tekrar kullanılan bir
depoya girer ve defalarca yeniden kullanılır. Senin gösterimlerin de öyle: onlar da sadece birkaç transition daha.

**2. Döngüde bir insan.** Kol işe yaramaz ya da tehlikeli bir şey yapmak üzereyken kontrolü devralırsın, düzeltirsin
ve kontrolü geri verirsin. Düzeltmen veri olarak saklanır. Keşif rastgele olmaktan çıkar ve "dene, kurtarıl,
kurtarmayı öğren"e dönüşür.

**3. Öğrenilmiş bir reward.** Bir denemenin başarılı olup olmadığını birinin söylemesi gerekiyor. İlk çalıştırmalarda
bir tuşa basmak yeter; ondan sonra başarı ve başarısızlık kareleriyle eğitilmiş küçük bir görüntü sınıflandırıcı bunu
otomatik yapar.

Bunlar bir arada **HIL-SERL** eder (Luo ve ark., 2024); LeRobot bunu uyguluyor.

## Nasıl bağlanıyor

```
           ┌──────────── Learner (GPU) ────────────┐
           │  SAC policy'yi replay buffer'lardan   │
           │  günceller                            │
           └───────▲───────────────────┬──────────┘
     transition'lar│                   │ yeni policy ağırlıkları, birkaç saniyede bir
           ┌───────┴───────────────────▼──────────┐
           │  Actor (kolun yanında)               │
           │  policy'yi çalıştırır, olanı gönderir │◄── sen: devral, geri ver
           └───────────────────────────────────────┘
                reward sınıflandırıcı: başarı? → reward 1, episode biter
```

gRPC üzerinden konuşan iki süreç: policy'yi kolda çalıştıran actor ve onu eğiten learner. Learner eğitirken actor kolu
hareket ettirmeyi hiç bırakmaz; taze ağırlıklar birkaç saniyede bir gelir (varsayılan 4 sn; dokümanlar 1 ila 2
öneriyor).

## İş akışı

Beş adım, her birinin kendi komutu var.

| | Adım | Komut |
| --- | --- | --- |
| 1 | Kolu görev alanında gezdirerek çalışma alanı sınırlarını bul | `lerobot-find-joint-limits` |
| 2 | Bir başarı tuşuyla yaklaşık 15 gösterim kaydet | `python -m lerobot.rl.gym_manipulator --config_path rl/env_config.json` |
| 3 | Her kamerayı önemli olan bölgeye kırp | `python -m lerobot.rl.crop_dataset_roi` |
| 4 | Başarı sınıflandırıcısını eğit | `lerobot-train --config_path reward_classifier_train_config.json` |
| 5 | Eğit: önce learner'ı, sonra actor'ı başlat | `python -m lerobot.rl.learner`, `python -m lerobot.rl.actor` |

## Bizim kayıt düzenimize neden hiç benzemiyor

| | Bizim taklit düzenimiz | Dokümanlardaki HIL-SERL |
| --- | --- | --- |
| Action uzayı | 6 eklem açısı | End-effector konumu; dokümanlara göre bazı görevleri "eklem uzayında öğrenmek neredeyse imkânsız" |
| Ne gerekiyor | Ekstra bir şey yok | Bir URDF ve ters kinematik; gripper'ın dışına çıkamayacağı bir sınır kutusu |
| Hız | 30 fps | Her örnek config'de 10 fps |
| Görüntüler | 640 × 480, bir VLM'e | Kırpılmış, sonra 128 × 128 ya da 64 × 64, küçük bir ResNet'e |
| Dil | 12 cümle | Yok: tek policy, tek görev |
| Görev süresi | 30 sn'ye kadar | "5-10 saniyede tamamlanabilen görevler" |
| Hesaplama | Kiralık, sonradan | Oturum boyunca çalışan bir NVIDIA GPU |

Yani RL, VLA'nın yerini almıyor. Görevin *kısa ve zor tek bir parçasına* oturuyor; taklidin sürekli yanlış yaptığı
parçaya.

## Bu hücrede ne olurdu

**Beceri.** "3. cepteki parçayı kavra ve 5 cm kaldır." Kısa, görsel olarak basit ve bir milimetrenin önemli olduğu
adım.

**Reward.** Üst kamera her cebin ve kasanın nerede olduğunu zaten hesaplıyor. "3. cep boş ve tepsinin üstünde hiçbir
şey durmuyor", eğitim gerektirmeyen bir başarı dedektörü. Bu, hücrenin çıplak bir masaya göre avantajı.

**Intervention'lar.** Kontrolü devraldığın müdahaleler için SO-101 leader iş görüyor ve dokümanlar onu SO-100'ünkine
tercih ediyor: dişlili eklemleri keşif sırasında robotu takip etmesini sağlıyor, böylece devralmak yumuşak oluyor.
Tuşlar: devralmak ve geri vermek için space, başarı için `s`, başarısızlık için `esc`.

**Dürüst maliyetler.** Kolla aynı ağda bir CUDA GPU; herhangi bir öğrenmeden önce sınırları, kırpmaları ve
sınıflandırıcıyı ayarlamak için bir öğleden sonra; birkaç yeniden başlatma, çünkü dokümanlar öğrenmenin "intervention
stratejisine çok duyarlı" olduğu konusunda uyarıyor.

Nasıl müdahale edeceğin, dokümanlardan: önce policy'nin birkaç episode boyunca keşfetmesine izin ver; kontrolü uzun
süre elinde tutma; sonrasında kendini kavrama gibi hızlı düzeltmelerle sınırla. İzlenecek sayı **intervention rate**,
yani ne sıklıkta müdahale ettiğin. İstikrarlı biçimde düşmeli. Düşmüyorsa policy senden öğrenmiyor.

## Reward model'ler: eğitmek zorunda olmadığın bir hakem

Bir başarı sınıflandırıcısı tek bir görevi bilir. Güncel LeRobot, bir videoyu bir talimata göre puanlayan üç genel
*reward model* ile geliyor.

| Model | Nasıl çalışıyor | Çıktı | Senin verinle eğitiliyor mu? |
| --- | --- | --- | --- |
| TOPReward | Dondurulmuş 8 B'lik bir VLM'e talimatın tamamlanıp tamamlanmadığını sorar ve "True" olasılığını okur | Klip başına tek puan | Hayır, zero-shot |
| ROBOMETER | Yörüngeleri puanlamak için fine-tune edilmiş 4 B'lik bir VLM | Kare başına 0'dan 1'e ilerleme ve bir başarı olasılığı | Hayır, LeRobot'ta zero-shot |
| SARM | CLIP özellikleri üzerinde, *senin* görevinin aşamalarını öğrenen küçük bir model | Aşama ve 0'dan 1'e ilerleme | Evet |

Üç kullanım, en ucuzu önce:

1. **Denemeleri puanla.** Modeli her rollout'un videosunda çalıştır ve kendi kararınla karşılaştır. Uyuşuyorlarsa
   sonraki her deney için otomatik bir hakemin var.
2. **Gösterimlerini puanla.** Duraklayan ya da geriye giden bir ilerleme eğrisi tereddütlü bir episode'u işaret eder.
   Bu, [gözden geçir](review.md) sayfasındaki "her episode'u izle" adımının bir model tarafından yapılmış hali.
3. **Eğitim verisini ağırlıklandır.** *Reward-aligned behaviour cloning* (RA-BC) aynı policy'yi aynı veriyle eğitir
   ama ilerleme kaydedilen karelere daha fazla ağırlık verir:

```bash
lerobot-train --reward_model.type=sarm --reward_model.annotation_mode=single_stage ...   # 1. fit SARM
python -m lerobot.rewards.sarm.compute_rabc_weights --dataset-repo-id ... --reward-model-path ...   # 2. score every frame
lerobot-train --policy.path=... --sample_weighting.type=rabc --sample_weighting.kappa=0.01 ...   # 3. train
```

RA-BC; SmolVLA, π0, π0.5 ve FastWAM için destekleniyor. Dokümanlar bizim için de geçerli bir uyarı ekliyor: dataset
zaten temizse pek faydası olmaz. Karşılığını DAgger turlarından sonra verir; veri iyi episode'ların, toparlanmaların
ve tereddütlerin bir karışımı olduğunda.

## RL, diğer iyileştirme yolları arasında nerede duruyor

| Yöntem | İnsan emeği | Sana ne kazandırır |
| --- | --- | --- |
| Daha fazla gösterim | Yüksek, her seferinde | Daha fazla kapsama, aynı tavan |
| DAgger düzeltmeleri | Orta, hedefli | Policy'nin gerçekten vardığı durumlar |
| RA-BC | Kurulumdan sonra yok | Elindeki veriden daha fazlası |
| HIL-SERL | Başta yüksek, giderek düşen | Kendi becerinin ötesi, tek bir kısa görevde |

Bunları bu sırayla yap. Her biri bir sonrakinden ucuz ve bir sonrakine gerek olup olmadığını söylüyor.

## Masaya ne götürmeli

- RL'in bir reward'a ihtiyacı var. Bu hücrede üst kamera bunu sağlayabilir; önce onu kur
  ([6. faz](roadmap.md#phase-6-better-eyes)), çünkü her deney otomatik bir hakemden faydalanır.
- RL görevini on saniyenin altında tut.
- Zero-shot bir reward model'i kaydedilmiş videolarda denemek bir saate mal olur. Bunu HIL-SERL'den çok önce yap.

**Kaynaklar:** [HIL-SERL](https://arxiv.org/abs/2410.21845) · LeRobot dokümanları:
[Train a robot with RL](https://huggingface.co/docs/lerobot/hilserl), [SARM](https://huggingface.co/docs/lerobot/sarm),
[ROBOMETER](https://huggingface.co/docs/lerobot/robometer), [TOPReward](https://huggingface.co/docs/lerobot/topreward) ·
[SARM makalesi](https://arxiv.org/abs/2509.25358) · [TOPReward makalesi](https://arxiv.org/abs/2602.19313)
