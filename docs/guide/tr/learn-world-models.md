# Policy olarak world model: Cosmos 3 Edge

SmolVLA sahneye bakar ve action'ları yazar. Bir *world action model* ise sahneye bakar ve aynı anda iki şey yazar:
action'ları ve o action'lar çalışırken kameraların göreceği videoyu. Bu sayfa bunun neden 10 kat parametreye
değdiğini, [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)'in içinde ne olduğunu ve bizim
dataset'imizin ondan tam olarak nasıl geçeceğini anlatıyor.

Bu sayfadaki hiçbir şey henüz bizim kolumuzda çalıştırılmadı. Sayılar model kartlarından ve
[LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3)'tan; bir seçim bize aitse öyle yazıyor.

## Neden video tahmin edelim ki

Bir gösterim karesi policy'ye öğrenmesi için altı sayı verir: eklem hedefleri. Aynı kare, video olarak, binlerce
sayıdır ve bu sayılar fiziği taşır: çeneler kapanana kadar parça kıpırdamaz, kol tepsiyi örter, düşürülen parça düşer.
Sonraki kareleri tahmin etmek zorunda olan bir model bunların hepsini temsil etmeye mecbur kalır ve bunu içinde robot
olmayan sıradan videodan öğrenebilir.

Yani bahis şu: büyük ölçekte video üzerinde önceden eğit, sonra action'ları tahmin edilecek bir şey daha olarak ekle.
Cosmos 3 yaklaşık 1,3 milyar örnekle eğitildi; bunların 348 M'si video, sadece 7 M'sinde action var.

## İçinde ne var

```
"put the part from pocket 3 in the REJECT bin"            top | wrist  (one canvas, current frame)
        │                                                        │
        ▼                                                        ▼ Wan2.2 VAE
┌─────────────────────┐      shared        ┌───────────────────────────────────────┐
│ understanding path  │◄──── attention ───►│ generation path (trained, flow matching)│
│ text tokens, causal │                    │ video latents + action tokens, noisy   │
└─────────────────────┘                    └───────────────────────────────────────┘
                                                 │                    │
                                                 ▼                    ▼
                                        future frames (latents)   next N actions
```

- **Bir Mixture-of-Transformers.** Her katmanın, her yol için bir tane olmak üzere iki ağırlık seti var; bunlar tek
  bir ortak attention'da buluşuyor. Edge'in 2048 genişliğinde 28 katmanı var: toplam 4,6 B parametre, varsayılan
  olarak 1,4 B'si eğitiliyor.
- **Tek tuval.** Kamera görüntüleri tek bir videoya döşenir: iki görüntü yan yana, üç görüntü biri üstte ikisi altta
  olacak şekilde. Modele hiçbir zaman "bu 2. kamera" denmedi; yerleşimi prompt'taki bir cümle tarif ediyor.
- **Birlikte denoising.** Mevcut kare temiz tutulur. Gelecekteki video latent'leri ve action chunk gürültü olarak
  başlar ve [flow matching](learn-flow.md) ile, UniPC solver'ı kullanılarak birlikte denoise edilir.
- **Prompt görevden daha uzun.** Talimatı, kare hızını, tuval boyutunu, yerleşim cümlesini ve chunk'taki boşta geçen
  kare sayısını taşır.

## SmolVLA ve Cosmos 3 Edge, yan yana

| | SmolVLA | Cosmos 3 Edge |
| --- | --- | --- |
| Parametre (eğitilen) | 450 M (~100 M) | 4,6 B (1,4 B; LoRA ile daha az) |
| Ne tahmin ediyor | 50 action | Action'lar ve gelecekteki video |
| Önceden eğitim | 481 topluluk robot dataset'i, çoğu SO-100 | Web ölçeğinde video ve görüntü; Franka, UR, WidowX, UMI, AgiBot ve başkalarından robot verisi |
| SO-100/101 gördü mü | Evet | Model kartında listelenmiyor |
| Fine-tuning belleği | Tek bir tüketici GPU'su | LoRA ile 21 GB, tam eğitimde 47 GB (ikisi de gradient checkpointing ile) |
| Tek karar | Kolun yanındaki Mac'te çalışacak kadar hızlı | NVIDIA'nın DROID benchmark'ında bir H100'de 1,25 sn |
| Nerede çalışıyor | macOS, Linux | Linux; sadece bfloat16 |
| LeRobot | 0.6.1 | 0.7, `cosmos3` extra'sı |
| Lisans | Apache 2.0 | OpenMDW 1.1 (ticari kullanıma izin var) |

Bizim için en önemli satır dördüncüsü. SmolVLA tam da bu kol için bir prior ile başlıyor. Cosmos 3 ise *dünyanın nasıl
hareket ettiğine* dair bir prior ile ve bizim robotumuz için eğitilmemiş action modülleriyle başlıyor: dokümanlar
bunları `--policy.load_action_modules=false` ile sıfırdan başlatmanı söylüyor.

## Bizim dataset'imiz onun içinden

Kaydımız gereksinimlere zaten uyuyor: LeRobot format v3.0, iki kamera, 6 eklem hedefi, kare başına bir cümle.

| Gereksinim | Bizimki | Ayar |
| --- | --- | --- |
| 1 ila 3 kamera, yerleşim sırasıyla | `top`, `wrist` | Yan yana tuval |
| Tuvali tarif eden bir cümle | Elle yazılmak zorunda; yanlış olanı fark edilmez | `--policy.view_description` |
| Gerçek kontrol hızı | 30 fps | Dataset'ten okunur |
| Action genişliği ≤ 64 | 6 | Dataset'ten okunur |
| `q01` / `q99` istatistikleri | `lerobot-record` yazıyor | Varsayılan `QUANTILES` normalizasyonu |
| Embodiment satırı | Herhangi biri; bütün satırlar eğitilmemiş başlıyor | `--policy.domain_id=0` |

İlk çalıştırma, tek GPU'da LoRA (flag'ler LeRobot dokümanlarından; chunk boyutu ve yerleşim cümlesi bizim seçimimiz):

```bash
pip install 'lerobot[cosmos3,training,peft]'      # Linux; install LeRobot from source if your release has no cosmos3 extra

lerobot-train \
  --policy.type=cosmos3_edge \
  --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.domain_id=0 \
  --policy.chunk_size=48 --policy.n_action_steps=48 \
  --policy.view_description='The left half shows a fixed camera looking down at the tray and the two bins; the right half shows the wrist-mounted camera.' \
  --peft.method_type=lora --peft.r=16 --peft.lora_alpha=32 \
  --policy.optimizer_lr=5e-4 --policy.optimizer_weight_decay=0 \
  --accelerator.mixed_precision=bf16 --tf32=false \
  --ema.type=power --ema.sigma_rel=0.1 \
  --batch_size=8 --steps=16000 \
  --dataset.repo_id=fport/so101-qc \
  --policy.repo_id=fport/so101-qc-cosmos3-edge
```

EMA ağırlıklarını (`checkpoints/last/pretrained_model_ema`) `--policy.dtype=bfloat16` ile değerlendir.

## Ters gitme olasılığı en yüksek üç şey

**1. Zaman.** NVIDIA, DROID policy'si için bir H100'de karar başına 1,25 sn ölçüyor (15 fps'te 32 action, 4 solver
adımı). Bizim 30 fps'imizde 32 action'lık bir chunk 1,07 sn'lik hareket eder: kol, sıradaki chunk daha ortada yokken
chunk'ı bitirir ve her sınırda durur. Ayarlanacak iki şey var: daha uzun bir chunk (48 action = 1,6 sn, yukarıdaki seçim) ve az
sayıda solver adımı (LIBERO varsayılanı 30, DROID tarifi 4 kullanıyor). İkisini de ayarlaman gerekeceğini bekle.

**2. Nerede çalıştığı.** bfloat16'da 15 GB GPU belleği ve sadece Linux, yani kolun yanındaki Mac'te değil. Policy bir
GPU makinesinde çalışmak ve chunk'ları geri göndermek zorunda. LeRobot bu ayrımı `lerobot.async_inference` olarak
sunuyor (bir policy sunucusu ve bir robot istemcisi). Ağ gecikmesi de 1. maddenin üstüne eklenir.

**3. Veri ölçeği.** Yayımlanmış tarifler 379 ila 1 693 episode üzerinde, 2 048'lik efektif batch ile fine-tune ediyor.
Bizde yaklaşık 120 episode var ve 8'lik bir batch kullanırdık. Bu ölçekte ne olduğunu kimse yayımlamadı. LIBERO
sonucunun kendisi (Edge ortalama %97,95, Nano %97,90) küçük modelin orada hiçbir şey kaybetmediğini söylüyor;
denenecek olanın Nano değil Edge olması bu yüzden.

## Tahmin edilen video bir hata ayıklama aracı

`--policy.save_predicted_video=true`, modelin her chunk için ne görmeyi beklediğini decode eder. Bu rehberdeki başka
hiçbir policy sana planını gösteremez. Kullan:

- Tahmin edilen video kolun yanlış cebe gittiğini gösteriyor: model cümleyi yanlış okumuş. Daha fazla dil çeşitliliği
  ya da cümle başına daha fazla veri.
- Tahmin edilen video doğru ama kol başka bir şey yapıyor: zayıf kısım action modülleri. Daha uzun eğit ya da
  normalizasyonu kontrol et.
- Tahmin edilen video bulanık ya da parça kayboluyor: sahne, modelin hayal edebildiğinin dışında. Önce tuval boyutunu
  ve yerleşim cümlesini kontrol et.

## Masaya ne götürmeli

- Bunu ancak SmolVLA'nın sabit testte bir skoru olduktan sonra çalıştır. Yoksa kıyaslayacak bir şey olmaz.
- Dürüst soru dar: *aynı 120 episode ile bir world model cep numarasını daha iyi izliyor mu?*
- Bütçe: eğitim için 48 GB ya da 80 GB'lık bir GPU ve rollout için koldan erişilebilen bir GPU makinesi.

**Kaynaklar:** [LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3) ·
[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) ve
[lerobot/Cosmos3-Edge](https://huggingface.co/lerobot/Cosmos3-Edge) model kartları ·
[Cosmos 3 technical report](https://arxiv.org/abs/2606.02800)
