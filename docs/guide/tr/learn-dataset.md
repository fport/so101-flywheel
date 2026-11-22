# Dataset'in içi

Programın kendisi dataset. Bu sayfa LeRobot'un yazdığı klasörü açıyor ve içinden tek bir eğitim örneğini takip ediyor.

## Diskte (format v3.0)

```
outputs/datasets/so101-qc/
  meta/info.json                         fps, feature shapes, episode and frame counts
  meta/stats.json                        mean, std, min, max, quantiles per feature
  meta/tasks.parquet                     the 12 sentences, numbered
  meta/episodes/chunk-000/file-000.parquet   one row per episode: its frames, its sentence, where its video starts
  data/chunk-000/file-000.parquet        one row per frame: state, action, timestamp, indices
  videos/observation.images.top/chunk-000/file-000.mp4
  videos/observation.images.wrist/chunk-000/file-000.mp4
```

Birçok episode aynı video dosyasını paylaşıyor. Bir episode satırı, o dosyada hangi saniyede başladığını saklıyor;
bir kare *başlangıç + kendi timestamp'i* noktasında bulunuyor. Video AV1: iki kameralı 30 sn'lik bir episode yüzlerce
değil, birkaç megabayt.

## Tek bir satır

| Sütun | Boyut | Ne |
| --- | --- | --- |
| `observation.state` | 6 | Follower'ın eklem konumları, derece cinsinden (gripper % olarak) |
| `action` | 6 | Aynı andaki leader eklem konumları: follower'a gönderilen komut |
| `observation.images.top`, `.wrist` | 480 × 640 × 3 | İhtiyaç olunca mp4'ten çözülüyor |
| `timestamp`, `frame_index`, `episode_index`, `task_index` | skaler | Kayıt tutma |

`action` "follower'ın sonra yaptığı şey" değil. Leader'ı tuttuğun yer. Follower birkaç kare geriden gelir ve policy
bu gecikmeyi hesaba katarak komut vermeyi öğrenir.

## Tek bir eğitim örneği

```
pick a random frame i (any episode)
observation  = images and state at frame i, plus that episode's sentence
target       = actions at frames i, i+1, … i+49
```

Bir episode'un sonuna yakın 50'den az action kalır. Sonuncusu tekrarlanır ve o adımlar `action_is_pad` olarak
işaretlenir, böylece loss'a sayılmazlar. **Her episode'u başlangıç pozunda bitirmenin** önemi bu yüzden: son gerçek
action'lar policy'ye bir görevin nasıl bittiğini öğretiyor.

## Normalizasyon: `stats.json` neden önemli

Birinci eklem 180° boyunca hareket ediyor, gripper %100 boyunca. Ham verilirse büyük sayılar loss'a hâkim olur. O
yüzden state ve action her boyut için ayrı normalize ediliyor:

```
x_normalised = (x − mean) / std
```

Policy normalize edilmiş action'lar tahmin ediyor ve bunlar motorlara gönderilmeden önce geri çevriliyor. İki sonucu
var:

- Verinde neredeyse hiç kıpırdamayan bir eklemin std'si çok küçüktür ve üzerindeki küçük gürültü büyür. Gösterimleri
  bütün kolu kullanarak yap.
- **Kayıttan sonra yeniden kalibre etmek policy'yi sessizce bozar.** Aynı poz artık farklı sayılarla okunur, yani
  normalize edilmiş her değer kayar.

## Zamanlama da verinin parçası

Bir kare ancak görüntü ile eklem okuması aynı ana aitse işe yarar. Kitaptaki örnek: t = 1,50 sn'deki bir kareyi
t = 1,58 sn'deki eklemlerle eşleştirmek yanlış eşlemeyi öğretir ve hata her episode'da tekrarlanır.

Burada neler ters gidebilir ve nasıl görürsün:

| Sorun | Sebep | Kontrol |
| --- | --- | --- |
| Düşen ya da geç gelen kareler | CPU meşgul (canlı bir görüntüleyici, bir tarayıcı, video encoding) | Görüntüleyici açmadan ve ağır hiçbir şey çalışmazken kaydet; bir episode'u tekrar oynat ve takılmaya bak |
| Bir kamera geriden geliyor | USB hub bant genişliği, düşük ışıkta otomatik pozlama | `scripts/review.sh N` içinde gripper iki görüntüde de aynı karede kapanmalı |
| Hiç kıpırdamayan bir eklem | Okunmamış bir servo | `observation.state` eğrisi düz |
| Follower takip etmiyor | Düşük besleme voltajı, aşırı yük | `action` ve `observation.state` eğrileri birbirinden uzaklaşıyor |

## Görevler sadece string

`--dataset.single_task` `tasks.parquet` içinde bir kez saklanıyor ve her kareden index'iyle referans veriliyor. Başka
bir cümleyle yeni bir blok kaydetmek o tabloya bir satır ekliyor. Bir cümleyi yanlış yazdıysan sonradan düzelt:

```bash
uv run lerobot-edit-dataset --repo_id=you/so101-qc --root=outputs/datasets/so101-qc \
  --new_root=outputs/datasets/so101-qc --operation.type=modify_tasks \
  --operation.episode_tasks='{"12": "put the part from pocket 3 in the OK bin"}'
```

Model cümleyi bir tokenizer üzerinden okuyor, yani `pocket 3` ile `pocket three` farklı girdiler. Tam olarak bir iki
token'la ayrışan on iki cümle temiz bir deney: policy neyi farklı yapıyorsa o token'lar yüzünden yapıyor.

**Kaynaklar:** LeRobot 0.6.1 kaynak kodu (`datasets/`, `processor/normalize_processor.py`) ·
[LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) · *Vision Language Models*, böl. 4
s. 147–160 ve böl. 11 s. 506–507
