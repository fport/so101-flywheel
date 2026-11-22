# SO-101: sıfırdan zirveye

Boş bir masadan, konuşur gibi verilen bir komutla parçaları ayıran bir robot kola. Kolu, kendi kaydettiğin veriyle
eğittiğin bir policy yönetiyor. Sekiz adım. Her biri birkaç sade `lerobot-*` komutu; arkasındaki sayfa komutun ne
yaptığını ve neye dikkat etmen gerektiğini anlatıyor.

**Görev.** Bir tepside 3D baskı altı parça var. Yukarıdaki bir kamera bir kez bakıyor ve her cep için OK ya da REJECT
kararı veriyor. Sonra kola her seferinde tek parça için şu söyleniyor: *"put the part from pocket 3 in the REJECT bin"*
(3. cepteki parçayı REJECT kasasına koy).

| | Adım | Sen ne yapıyorsun | Ne öğreniyorsun |
| --- | --- | --- | --- |
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
| [Bir policy'yi ölçmek](learn-eval.md) | 8 | 24'te 18 sana gerçekte ne söylüyor, neyi not etmelisin |

Sonra: [sırada ne denenir](next.md) (Cosmos 3, sim-to-real aktarımı, grounding modelleri, RL) ve
[tek sayfada kavramlar](concepts.md).

## Neye ihtiyacın var

- SO-101 leader + follower kol, iki USB webcam (biri yukarıda, biri gripper'da)
- Hücre için bir 3D yazıcı, ya da masaya sabitlediğin herhangi bir tepsi ve iki kutu
- Kayıt için Mac ya da Linux bir laptop; eğitim için kiralık bir GPU (yaklaşık 4 saat A100, kabaca 10 $)
- Ücretsiz bir [Hugging Face](https://huggingface.co) hesabı: dataset de policy de orada duruyor

## Bugün, sırayla

```bash
scripts/teleop.sh --cams      # 1. both cameras see what they should
uv run flywheel-inspect       # 2. the top camera reads the tray
scripts/record.sh 1 ok        # 3. first block of 10 episodes
scripts/status.sh             # 4. what is left; repeat until 120
scripts/train.sh              # 5. SmolVLA on a rented GPU
scripts/rollout.sh 1 ok       # 6. the policy moves the arm
```
