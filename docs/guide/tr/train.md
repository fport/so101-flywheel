# 7 · Eğitim

**Hedef:** 450 M parametreli bir vision-language-action modeli olan [SmolVLA](https://huggingface.co/docs/lerobot/smolvla)'yı
120 episode'unla fine-tune etmek. Görmeyi ve okumayı zaten biliyor; sen ona kendi kolunu, hücreni ve 12 cümleni
öğretiyorsun.

```bash
scripts/train.sh
```

Bu, kiralık bir Hugging Face GPU'sunda şunu çalıştırır (dakika başına hesabına faturalanır; fiyatlar için
`hf jobs hardware`):

```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=fport/so101-qc --policy.repo_id=fport/so101-qc-smolvla \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --policy.device=cuda --job.target=a100-large
```

| Flag | Neden |
| --- | --- |
| `--policy.path=lerobot/smolvla_base` | Önceden eğitilmiş modelden başla, sıfırdan eğitme |
| `--rename_map` | Önceden eğitilmiş model kameralarına `camera1..3` diyor; bizimkiler `top` ve `wrist`. Çalıştırırken de yine lazım |
| `--steps=20000` | LeRobot'un dokümanlardaki tarifi: A100'de yaklaşık 4 saat |
| `--batch_size=64` | GPU'nun belleği yetmezse düşür |
| `--job.target=a100-large` | Hangi kiralık GPU: saati 2,50 $, yani bu eğitim yaklaşık 10 $. Kendi makinende eğitmek için bu flag'i hiç verme |

**Onun yerine Colab:** resmi
[SmolVLA notebook'unu](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
aç, L4 ya da A100 runtime seç ve aynı flag'leri `--job.target` olmadan kullan. Ücretsiz T4 bu batch size için fazla
küçük.

## İçeride ne oluyor

1. Her kamera karesi yaklaşık 64 token'a dönüşür, cümle token'lara dönüşür, eklem açıları da bir token daha olur.
2. Dondurulmuş vision-language modeli bunları bir kez okur.
3. Küçük bir *action expert* (yaklaşık 100 M parametre; yalnızca o ve birkaç küçük giriş/çıkış katmanı eğitilir) gürültüyü 10 denoising adımında sonraki
   50 action'a çevirir. Bu *flow matching*.
4. Loss, tahmin edilen 50 action'ın senin gerçekten yaptığın 50 action'dan ne kadar uzak olduğu. Başka hiçbir şey
   değil: ödül yok, başarı diye bir kavram yok.

Formülleriyle birlikte resmin tamamı: [bir vision-language modelin içi](learn-vlm.md) ve [SmolVLA'nın içi](learn-vla.md).

Yani düşen loss "kaydedilen hareketleri daha iyi taklit ediyor" demek, "parçaları ayırıyor" değil. Onu sana sadece
gerçek kol söyler.

## Kontrol

- Loss ilk bin adımda hızlı, sonra yavaş düşüyor. Baştan beri düz gidiyorsa dataset ya da kamera eşlemesi yanlış.
- Job başlarken **Model repo:** `fport/so101-qc-smolvla_<zaman>` yazar. Eğitim bitince policy oraya (private) yüklenir.
  Bu adı `scripts/env.local.sh` içine `POLICY=...` olarak yaz: [8. adım](rollout.md) oradan okur.

## Önce kısa bir deneme yap

İlk iki bloktan sonra `STEPS=200 scripts/train.sh` çalıştır. İşe yarar bir policy üretmez; ama iki saatlik veri
kaydetmeden önce dataset'in yüklendiğini, kamera adlarının eşlendiğini ve upload'un çalıştığını kanıtlar.
