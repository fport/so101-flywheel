# SmolVLA'nın içi

SmolVLA 450 M parametre: sahneyi okuyan küçük bir VLM ve hareketi yazan 100 M'lik bir *action expert*. Bu sayfa bir
eğitim adımını ve bir kontrol adımını, bu projenin gerçek sayılarıyla, modelin içinden takip ediyor.

```
top 640×480 ─┐                                   ┌──────────────────────────┐
wrist 640×480├─► SigLIP + pixel shuffle ─► 128 ─►│ SmolVLM2, first 16 layers │──► keys/values (cached)
"put the part from pocket 3…" ─► tokens ─► 48 ──►│        (frozen)           │          │
6 joint angles ─► pad to 32 ─► linear ─► 1 ─────►└──────────────────────────┘          ▼
                                                 ┌──────────────────────────┐   cross-attention
noise (50 × 32) ─► + time embedding ─► 50 ──────►│  action expert, 16 layers │◄─────────┘
                                                 │        (trained)          │──► 50 actions × 6 joints
                                                 └──────────────────────────┘
```

## İçeri ne giriyor

| Girdi | İşleme | Token |
| --- | --- | --- |
| Her kamera karesi | 512 × 512'ye sığacak şekilde boyutlandırılır (bizimkiler 512 × 384 olur, dolgu eklenir), [−1, 1]'e ölçeklenir, SigLIP, pixel shuffle | 64 |
| Cümle | Sonuna bir satır sonu eklenir, token'lara ayrılır, 48'e doldurulur | 48 |
| Eklem açıları | Dataset'in ortalaması ve std'siyle normalize edilir, 6'dan 32'ye sıfırla doldurulur, tek bir lineer katman | 1 |

Yani VLM karar başına **177 token** okuyor. Önceden eğitilmiş model kameralarına `camera1..3` diyor; `--rename_map`
`top` ve `wrist`'i ilk ikisine eşliyor, üçüncüsü de basitçe dışarıda kalıyor.

Tek bir checkpoint'in farklı eklem sayısına sahip kollarda çalışabilmesi 32'ye doldurma sayesinde.

## Yan yana iki transformer

- **VLM**, üst yarısı atılmış SmolVLM2-500M: 32 dil katmanının sadece ilk 16'sı kullanılıyor. Kontrol için orta
  katman özelliklerinin yettiği ortaya çıkıyor; katmanların yarısı zamanın yarısı demek.
- **Action expert**'in de 16 katmanı var, genişliği 0,75 kat. Katmanları sırayla değişiyor: çift katmanlar her şeye
  birlikte attention yapıyor, tek katmanlar *cross-attention* yapıyor, yani action token'ları VLM'in önbelleğe alınmış
  key ve value'larına bakıyor.

Bir karar sırasında VLM **bir kez** çalışıyor. Çıktısı önbelleğe alınıyor ve expert onu on denoising adımının hepsinde
yeniden kullanıyor. Bir VLA'yı yeterince hızlı yapan şey bu önbellek.

## Flow matching: gürültü nasıl harekete dönüşür

Expert action'ları doğrudan vermiyor. Bir *velocity field* (hız alanı) öğreniyor: yarı gürültülü herhangi bir action
chunk için hangi yönün temiz bir chunk'a götürdüğünü söyleyen bir kural.

**Eğitim**, tek bir örnek için (rastgele bir kare ve onu izleyen 50 action):

```
A  = the 50 real actions                    (50 × 6, normalised; zero-padded to 50 × 32)
ε  = random noise of the same shape
t  = a random number in (0, 1], biased toward 1 (the noisy end)

x_t = t·ε + (1 − t)·A          # mix: t = 1 is pure noise, t = 0 is the real chunk
u   = ε − A                    # the direction from data to noise
loss = mean( (expert(x_t, t, scene) − u)² )
```

Ağ, senin yaptığının bulanık bir hâlini görüyor ve bulanıklığın hangi yöne baktığını söylemek zorunda. Episode'un
sonunu aşan adımlar loss'tan maskeleniyor.

**Inference** kuralı saf gürültüden geriye doğru, on Euler adımında çalıştırıyor:

```
x = ε                           # start: noise
for t in 1.0, 0.9, … 0.1:
    x = x − 0.1 · expert(x, t, scene)
return x                        # 50 actions
```

Neden action'ları doğrudan regresyonla tahmin etmiyoruz? [Imitation learning nasıl çalışır](learn-imitation.md)
sayfasındaki "birden çok doğru cevap" sorunu yüzünden: regresyon iki geçerli kavramanın ortalamasını alıp geçersiz
bir kavrama üretir. Flow matching rastgele bir ε'dan başlayıp gösterilen davranışlardan *birinde* biter; farklı bir
ε için farklı birinde.

## 50 action'dan hareket eden bir kola

| | |
| --- | --- |
| Chunk | 50 action = 30 fps'te 1,67 sn |
| Varsayılan (`--inference.type=sync`) | 50'sini de oynat, sonra dur ve düşün. Model yavaşsa kol her chunk sınırında duraklar |
| Real-time chunking (`rtc`, `scripts/rollout.sh` bunu kullanıyor) | 30 action kaldığında arka plandaki bir thread sıradaki chunk'ı başlatır |

RTC'nin bir inceliği var. Yeni chunk, geldiği anda zaten eskimiş bir gözlemden hesaplanıyor. Bu yüzden yeni chunk
denoise edilirken ilk action'ları eski chunk'tan kalanlara doğru çekiliyor: chunk gelene kadar zaten çalışmış olacak
adımlarda tamamen, 10. adıma kadar azalarak. O bayat adımlar sonra atılıyor. Sonuç: kesintisiz hareket ve her 20
action'da (0,67 sn) yeni bir plan; model bundan yavaşsa daha seyrek.

## Fine-tune neyi değiştiriyor

`lerobot-train --policy.path=lerobot/smolvla_base` vision encoder'ı ve VLM'i dondurulmuş tutuyor ve yaklaşık 100 M
parametre eğitiyor: action expert, ona giren ve ondan çıkan projeksiyonlar ve eklem durumunu gömen tek katman.

| Ayar | Değer | Anlamı |
| --- | --- | --- |
| Optimizer | AdamW, lr 1e-4 | Isınma, ardından 2.5e-6'ya cosine azalma |
| Batch × adım | 64 × 20 000 | 1,28 M örnek: bizim ~36 000 karemizin her biri yaklaşık 35 kez görülüyor |
| Normalizasyon | `meta/stats.json`'dan mean/std | Senin dataset'inin istatistikleri önceden eğitilmiş olanların yerini alıyor |
| Augmentation | kapalı | Işığa dayanıklılık senin verinden gelmek zorunda |

Model 481 topluluk dataset'iyle (22,9 bin episode, çoğu SO-100 kol) önceden eğitilmiş, kameralar top / wrist / side
olarak eşlenmiş. Senin üst ve bilek kamerası görüntülerinin iyi oturması ve cümle başına 10 episode'un yetebilmesi bu
yüzden.

## Diğerleriyle kıyas

| Policy | Boyut | Dil okuyor mu | Action head | Ne ister |
| --- | --- | --- | --- | --- |
| ACT | LeRobot'ta ~52 M (makalede 80 M) | hayır (LeRobot'ta) | CVAE transformer, 100'lük chunk'lar, L1 loss | Herhangi bir GPU, laptop bile |
| **SmolVLA** | 450 M | evet | Flow matching, ~100 M expert | Tek bir tüketici GPU'su |
| π0 / π0.5 | 3–4 B | evet | Flow matching, ~300 M expert | 24–40 GB |
| GR00T N1.7 | 3 B | evet | Daha yavaş bir VLM'in altında flow matching'li diffusion transformer | Büyük GPU, çalıştırmak için Jetson |
| Cosmos 3 Edge | 4,6 B | evet | Action'ları world model'in kendisi denoise ediyor | LoRA ile 21 GB, LeRobot 0.7 |

Her biri 50 gösterimli gerçek SO-100 görevlerinde SmolVLA makalesi ortalama %78 başarı bildiriyor; π0 için %62, ACT
için %48, önceden eğitimi olmayan SmolVLA için %52. Bizim görevimizde tek kelimeyle ayrışan 12 cümle var, bu da üç ayrı
görevden daha zor: ilk bozulacak şeyin cep numarası olmasını bekle.

**Kaynaklar:** LeRobot 0.6.1 kaynak kodu (`policies/smolvla/modeling_smolvla.py`, `smolvlm_with_expert.py`,
`policies/rtc/`) · [SmolVLA paper](https://arxiv.org/abs/2506.01844) ·
[Flow matching](https://arxiv.org/abs/2210.02747) · [π0](https://arxiv.org/abs/2410.24164) ·
[Real-time chunking](https://arxiv.org/abs/2506.07339) · *Vision Language Models*, böl. 11, s. 504–514
