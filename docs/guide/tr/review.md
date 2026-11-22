# 6 · Gözden geçir ve yükle

**Hedef:** GPU saatlerini harcamadan önce dataset'te ne olduğunu bilmek.

```bash
scripts/status.sh              # episodes per sentence
scripts/review.sh 12           # replay episode 12: both cameras and all joints, in Rerun
scripts/review.sh delete 12 40 # remove bad episodes; the previous version is kept as so101-qc_old
scripts/review.sh push         # upload after deleting: training reads the Hub copy, not your disk
```

Online, hiçbir şey kurmadan:
`https://huggingface.co/spaces/lerobot/visualize_dataset` adresini aç ve `fport/so101-qc` yaz (private bir dataset
için giriş yapmış olmalısın).

## Neye bakmalı

Her bloktan en az iki episode'u ve emin olamadığın her episode'u izle.

- İki video da oynuyor, siyah değil ve aynı anı gösteriyor (gripper ikisinde de aynı anda kapanıyor).
- `action` eğrisi (leader) ile `observation.state` eğrisi (follower) küçük bir gecikmeyle üst üste biniyor. Düz bir
  çizgi, okunmamış bir servo demek.
- Episode yaklaşık yarım saniye içinde hareket etmeye başlıyor ve başlangıç pozunda bitiyor.
- Kadrajda el yok, leader kol yok.

## Klasörde ne var

```
outputs/datasets/so101-qc/
  meta/info.json        fps, features (6 joints in, 6 out, two cameras), episode count
  meta/tasks.parquet    the 12 sentences
  meta/episodes/        which frames and which sentence belong to which episode
  meta/stats.json       mean/std per feature: used to normalise during training
  data/                 joint angles and actions, 30 rows per second (parquet)
  videos/               one mp4 stream per camera
```

Daha fazla ayrıntı: [dataset'in içi](learn-dataset.md).

Bir kare = `observation.images.top`, `observation.images.wrist`, `observation.state` (follower'ın nerede olduğu),
`action` (leader'ın ona nereye gitmesini söylediği) ve görev cümlesi. Eğitim şu: ilk üçü ve cümle verildiğinde
sonraki 50 action'ı tahmin et.
