# Sırada ne denenir

Bunların her biri net bir sorusu olan tek bir deney. Bunları SmolVLA baseline'ının bir puanı olduktan sonra yap:
baseline olmadan bir şeyin işe yarayıp yaramadığını anlayamazsın.

## Daha büyük bir policy: Cosmos 3 Edge

[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) 4 B parametreli bir *world model*: tek bir ağla
video, metin ve action üretiyor, yani SmolVLA gibi takıp çalıştırabileceğin bir VLA değil. LeRobot (0.7 ve sonrası)
onu bir policy olarak fine-tune edebiliyor:

```bash
lerobot-train --policy.type=cosmos3_edge --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false --dataset.repo_id=fport/so101-qc ...
```

- **Soru:** 10 kat büyük bir model, aynı 120 episode ile 12 cümleye daha iyi uyuyor mu?
- **Maliyet:** LoRA fine-tune yaklaşık 21 GB GPU belleği istiyor (A10G değil, A100); tam fine-tune 47 GB. Sadece Linux.
- **Dikkat:** model kartında SO-101 geçmiyor; NVIDIA'nın blogunda geçiyor. Beklediği kamera tanımı için
  [LeRobot Cosmos 3 sayfasını](https://huggingface.co/docs/lerobot/cosmos3) oku.

## Sim-to-real: Cosmos 3 Nano Sim-Transfer

[nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) simülatör çıktısını
(derinlik, segmentasyon ya da kenarlar) ve bir metin prompt'unu fotogerçekçi videoya çeviriyor. Bu repoda hücrenin bir
MuJoCo kopyası ve onu çözen script'li bir uzman var (`uv run flywheel-sim batch`).

- **Soru:** gerçek gibi görünsün diye yeniden boyanmış sim episode'ları, teleop ile kaydedilenlerin bir kısmının yerini
  tutabilir mi?
- **Maliyet:** 15–16 B parametre, NVIDIA sadece H100 sınıfı GPU'larda test etmiş. 24 GB'tan fazlasını bekle.
- **Yol:** sim'den derinlik + segmentasyon render et → Sim-Transfer → sim'in action'larıyla bir LeRobot dataset'i →
  gerçek + üretilmiş veriyle eğit → aynı 24 denemelik test.

## Grounding: LocateAnything {#grounding-locateanything}

[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) "gear with a missing tooth" (dişi eksik
dişli) gibi bir metin prompt'u için kutular ya da noktalar (0–1000 koordinat) döndürüyor.
[Demo](https://huggingface.co/spaces/nvidia/LocateAnything).

- **Soru:** bir prompt, `flywheel-inspect` içindeki elle yazılmış ölçümlerin ve AprilTag'lerin yerini tutabilir mi?
- **Test:** parça ID'leri bilinen `outputs/inspect/*/frame.png` kareleri hazır bir benchmark.
- **Dikkat:** ticari olmayan lisans; herkese açık demo yüklediklerini saklayabilir.

## Pekiştirmeli öğrenme (RL)

Taklit, gösterimlerinin kalitesinde durur. RL, kolun kendi denemelerinden ve bir başarı sinyalinden öğrenerek onları
aşmasını sağlar. LeRobot HIL-SERL ile geliyor (gerçek kolda human-in-the-loop RL). Ödülü muayene kamerası verebilir:
"parça komut verilen kasada mı?". Buna ancak taklit bir platoya ulaşınca başla.

## Daha yeni bir LeRobot

Bu repo LeRobot 0.6.1'e sabitlenmiş (kol onunla kalibre edildi ve test edildi). 0.7, Cosmos 3 policy'lerini ve uzaktan
inference'ı ekliyor. Dataset formatı (v3.0) aynı, yani 0.6.1'de kaydedip 0.7'de eğitebilirsin.
