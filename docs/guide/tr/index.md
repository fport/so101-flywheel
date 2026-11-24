# SO-101: sıfırdan zirveye

Boş bir masadan, konuşur gibi verilen bir komutla parçaları ayıran bir robot kola. Kolu, kendi kaydettiğin veriyle
eğittiğin bir policy yönetiyor. Sekiz adım. Her biri birkaç sade `lerobot-*` komutu; arkasındaki sayfa komutun ne
yaptığını ve neye dikkat etmen gerektiğini anlatıyor.

Bu sekiz adım daha uzun bir projenin ilk dört fazı. [Rota](roadmap.md) on fazın hepsini, her birinin ne ürettiğini ve
bittiğini nasıl anlayacağını gösteriyor.

**Görev.** Bir tepside 3D baskı altı parça var. Yukarıdaki bir kamera bir kez bakıyor ve her cep için OK ya da REJECT
kararı veriyor. Sonra kola her seferinde tek parça için şu söyleniyor: *"put the part from pocket 3 in the REJECT bin"*
(3. cepteki parçayı REJECT kasasına koy).

| | Adım | Sen ne yapıyorsun | Ne öğreniyorsun |
| --- | --- | --- | --- |
| 0 | [Önce simüle et](simulate.md) | `flywheel-sim batch`, donanım yok | Görev ve hücrenin neden böyle göründüğü |
| 1 | [Hücreyi kur](cell.md) | Bas, birleştir, iki kamerayı tak | Sahne neden kıpırdamamalı |
| 2 | [Kolu hazırla](setup.md) | Kur, kalibre et, teleop yap | Leader/follower, eklem uzayı |
| 3 | [Tepsiyi muayene et](inspect.md) | `flywheel-inspect` | Klasik görüntü işleme ve VLM, grounding |
| 4 | [Kayıt](record.md) | `scripts/record.sh 1 ok` × 12 blok | Gösterimlerden oluşan bir dataset nedir |
| 5 | [Veri kalitesi](data-quality.md) | Kayıt sırasında kontrol listesine uy | Policy'ler neden başarısız olur: sebep veri |
| 6 | [Gözden geçir ve yükle](review.md) | İzle, kötü episode'ları sil, push et | LeRobot dataset formatı |
| 7 | [Eğitim](train.md) | `scripts/train.sh` (SmolVLA, kiralık GPU) | VLA, action chunk, flow matching |
| 8 | [Çalıştır ve iyileştir](rollout.md) | `scripts/rollout.sh 3 ok`, puanla, düzelt | Değerlendirme, DAgger, veri çarkı (flywheel) |

## Ne yaptığını anla

Her adım bunlar olmadan da çalışır. Bunları, işin *neden* çalıştığını ve çalışmadığında neyi değiştireceğini bilmek
için oku.

| Oku | Hangi adımdan önce | Neyi cevaplıyor |
| --- | --- | --- |
| [Imitation learning nasıl çalışır](learn-imitation.md) | 4 | Küçük hatalar neden çığ gibi büyür, gösterimler neden tutarlı olmalı, chunk nedir |
| [Dataset'in içi](learn-dataset.md) | 6 | Tam olarak ne saklanıyor, tek bir eğitim örneği nasıl kuruluyor, zamanlama neden önemli |
| [Bir vision-language modelin içi](learn-vlm.md) | 7 | Bir görüntü nasıl 64 token oluyor, ne dondurulmuş, modeller *nerede* olduğunu nasıl söylüyor |
| [SmolVLA'nın içi](learn-vla.md) | 7 | İki transformer, altı satırda flow matching, real-time chunking |
| [Sıfırdan flow matching](learn-flow.md) | 7 | Gürültü nasıl harekete dönüşür, çalıştırabileceğin 50 satırlık bir örnekle |
| [Bir policy'yi ölçmek](learn-eval.md) | 8 | 24'te 18 sana gerçekte ne söylüyor, neyi not etmelisin |

Daha ileri gitmek için, baseline'ın bir puanı olduktan sonra:

| Oku | Neyi cevaplıyor |
| --- | --- |
| [2026'da alan](learn-landscape.md) | Beş policy ailesi ve hangilerinin bu kolu zaten gördüğü |
| [Policy olarak world model](learn-world-models.md) | Cosmos 3 Edge nedir, bizim dataset'imiz ondan nasıl geçer |
| [Sim-to-real](learn-sim2real.md) | Yeniden boyanmış bir simülatör neyin yerini tutabilir, neyin tutamaz |
| [Grounding](learn-grounding.md) | LocateAnything bir ifadeyi nasıl cep numarasına çevirir |
| [Gerçek kolda reinforcement learning](learn-rl.md) | HIL-SERL ve reward model'ler nasıl çalışır, bu hücreden ne isterler |
| [Deney planı](next.md) | Bunlardan hangisini önce denemeli, sonuçları nasıl karşılaştırmalısın |

Ve [tek sayfada kavramlar](concepts.md).

## Neye ihtiyacın var

- SO-101 leader + follower kol, iki USB webcam (biri yukarıda, biri gripper'da)
- Hücre için bir 3D yazıcı, ya da masaya sabitlediğin herhangi bir tepsi ve iki kutu
- Kayıt için Mac ya da Linux bir laptop; eğitim için kiralık bir GPU (yaklaşık 4 saat A100, kabaca 10 $)
- Ücretsiz bir [Hugging Face](https://huggingface.co) hesabı: dataset de policy de orada duruyor

## Bugün, sırayla

```bash
scripts/teleop.sh             # 1. leader drives follower; scripts/cams.sh shows both cameras
uv run flywheel-inspect       # 2. the top camera reads the tray
scripts/record.sh 1 ok        # 3. first block of 10 episodes
scripts/status.sh             # 4. what is left; repeat until 120
scripts/train.sh              # 5. SmolVLA on a rented GPU
scripts/rollout.sh 1 ok       # 6. the policy moves the arm
```
