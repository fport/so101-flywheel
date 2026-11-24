# 2026'da alan

Bu proje başladığında "küçük bir kol için policy" demek ACT ya da SmolVLA demekti. LeRobot'un Kasım 2026'daki
dokümantasyonu (0.7.0 sürümü ve ondan sonraki main branch'i) yirmiden fazlasını kapsıyor. Bu sayfa onları ailelere ayırıyor, gerçekte neyin farklı olduğunu söylüyor ve
bir SO-101'de çalıştırmaya değer olanları seçiyor.

Buradaki her şey, her policy'nin [LeRobot dokümantasyonu](https://huggingface.co/docs/lerobot)'ndan alındı.
Bir sayfa sayı vermiyorsa bu sayfa da vermiyor.

## Beş aile

Her policy aynı soruyu cevaplıyor: *gördüğüme ve bana söylenene göre, önümüzdeki saniyede ne yapacağım?*
Yolda neyi tahmin ettiklerinde ayrışıyorlar.

```
                     ┌─ 1. Küçük behaviour cloning ── sadece action'lar, dil modeli yok
                     ├─ 2. VLA, flow head ─────────── bir VLM okur; bir head bir action chunk'ı denoise eder
     gözlem ─► model ├─ 3. VLA, action token'ları ─── VLM action'ları metin gibi, token olarak yazar
+ talimat            ├─ 4. Akıl yürüten VLA ───────── önce bir alt görev (ya da kutular) yazar, sonra hareket eder
                     └─ 5. World-action model ─────── gelecekteki videoyu action'larla birlikte tahmin eder
```

### 1. Küçük behaviour cloning

| Policy | Boyut | Not |
| --- | --- | --- |
| ACT | dokümanlara göre ~80 M | Dil yok. Tek GPU'da birkaç saatte sıfırdan eğitilir. Hâlâ önerilen ilk model |
| Multitask DiT | ~450 M | CLIP ile koşullandırılmış bir diffusion transformer. 64+ batch ve, kendi dokümanlarına göre, 300'den fazla örnek ister |

Bunları taban olarak kullan. Tek cümlede ACT, o cümlede senin VLA'nla aynı sonucu veriyorsa VLA henüz pek bir şey katmıyor.

### 2. Flow matching head'li VLA

Bu rehberin üzerine kurulduğu aile: [SmolVLA'nın içi](learn-vla.md), [flow matching](learn-flow.md).

| Policy | Boyut | Onu ayıran şey |
| --- | --- | --- |
| SmolVLA | 450 M | Bir tüketici GPU'suna sığacak kadar küçük; topluluk SO-100 verisiyle önceden eğitilmiş |
| π0 | 3 B | Diğerlerinin kopyaladığı tasarım. 50 action'lık chunk'lar |
| π0.5 | belirtilmemiş | Açık dünyada genelleme için karışık veriyle birlikte eğitilmiş; eklem durumu prompt'a metin olarak yazılıyor |
| GR00T N1.7 | 3 B | NVIDIA. 16 action'lık chunk'lar, göreli action'lar. Dokümanlar "yeni embodiment" örneği olarak SO-101'i kullanıyor |
| X-VLA | 0,9 B | Robot başına öğrenilmiş bir *soft prompt*; uyarlamak ağırlıkların %1'ini ayarlamak anlamına gelebiliyor |
| EO-1 | 3 B | İç içe geçmiş görüntü, metin ve action üzerinde önceden eğitilmiş; soruları da cevaplayabiliyor |
| EVO1 | ~1 B VLM | İki eğitim aşaması: önce head, sonra VLM |
| WALL-OSS, Wall-OSS-0.5 | 4 B'ye kadar | Bir Qwen VLM'in üzerinde flow ya da token head |
| LingBot-VLA 2.0 | repo adına göre 6 B | Mixture-of-experts bir action head; derinlik ve video eğitim sırasında distile ediliyor |

### 3. Action'ları token olarak yazan VLA

| Policy | Boyut | Onu ayıran şey |
| --- | --- | --- |
| π0-FAST | PaliGemma, 2 B dil modeli | Bir chunk'ı DCT ve byte-pair encoding ile sıkıştırır, sonra metin gibi tahmin eder. π0'dan 5 kata kadar hızlı eğitilir |
| NAT VLA | 450 M | Token'lar zamana göre sıralı, bu yüzden action'lar üretilirken akıtılabiliyor |

Denoising döngüsü yok: dil modelinin tek bir geçişi chunk'ı yazıyor. Bedeli, senin action istatistiklerine uymak
zorunda olan bir tokenizer.

### 4. Akıl yürüten VLA'lar

Bunlar action'dan önce insanın okuyabileceği bir şey yazıyor.

| Policy | Önce ne yazıyor | Ne ister |
| --- | --- | --- |
| G0.5 | Mevcut alt görev, isteğe bağlı olarak bounding box'lar | Akıl yürütmenin eğitilmesi için alt görev (ve kutu) etiketleri |
| FineART-VLA | Görev → alt görev → action, isteğe bağlı bellekle | Alt görev etiketleri; onlar olmadan π0.5 gibi davranıyor |
| MolmoAct2 | "Think" varyantında derinlik üzerine akıl yürütme | Think varyantı henüz LeRobot'ta yok |

Güncel LeRobot'ta bunların dayandığı araçlar var: `lerobot-annotate` bir VLM'e episode'larını izletiyor ve alt görevleri
yeni dataset sütunlarına yazdırıyor; 30 saniyelik episode başına yaklaşık 50 VLM çağrısı.

Bunun bizim hücre için neden önemli olduğu: "put the part from pocket 3 in the REJECT bin" cümlesinin doğal alt görevleri var (uzan, kavra,
kaldır, taşı, bırak). Hangisinde olduğunu söyleyen bir policy, hata ayıklayabileceğin bir policy'dir.

### 5. World-action modeller

Gelecekteki videoyu ve action'ları birlikte tahmin ediyorlar: [policy olarak world model](learn-world-models.md).

| Policy | Boyut | Onu ayıran şey |
| --- | --- | --- |
| Cosmos 3 Edge / Nano | 4,6 B / 16,5 B | NVIDIA. Video ve action'lar birlikte denoise ediliyor; tahmin edilen video kaydedilebiliyor |
| FLUX 3 Action | belirtilmemiş | Black Forest Labs. Video latent'lerini ve action'ları tahmin ediyor, sadece action'ları döndürüyor. Bir SO-101 checkpoint'i ile geliyor |
| FastWAM | 5 B backbone | Eğitimde videodan öğreniyor, inference'ta action'ları doğrudan tahmin ediyor |
| Long-WAM | belirtilmemiş | NVIDIA. Bağlam olarak 2,4 sn'lik geçmiş |
| LingBot-VA | ~5 B | Video latent'lerini ve action'ları autoregressive olarak iç içe diziyor |
| VLA-JEPA | 2 B VLM | Geleceği *latent* uzayda, ve sadece eğitim sırasında tahmin ediyor |
| LaWAM | belirtilmemiş | Videodan öğrenilmiş latent action'lar |
| Praxis-1 | 27 B | Runway. Sadece çalıştırmak için bile 80 GB'lık bir GPU ister |

## SmolVLA'dan beri değişen dört şey

1. **Eğitim sinyali olarak gelecek.** Yeni modellerin yarısı videoyu ya da latent'lerini tahmin ediyor. Birkaçı (FastWAM,
   VLA-JEPA, FLUX 3 Action) bu tahmini inference'ta atıyor: orada olmasının tek sebebi ağa sahnelerin nasıl
   hareket ettiğini öğretmekti.
2. **Sesli düşünmek.** Ara çıktı olarak alt görevler ve kutular, kendi verini etiketlemen için araçlarla birlikte.
3. **Varsayılan olarak LoRA.** Modeller tek GPU'da tam fine-tune edilemeyecek kadar büyük, bu yüzden dokümanlar adapter'larla başlıyor.
4. **Policy robotun üstünden taşındı.** LeRobot eski async modülünü *remote inference* ile değiştirdi: bir GPU
   makinesinde policy sunucusu, kolun yanındaki bilgisayarda rollout. Büyük modeller bu olmadan bir masada kullanılamaz.

## Herkesin alıntıladığı benchmark

| Policy | LIBERO ortalaması |
| --- | --- |
| Long-WAM | %99,0 |
| LaWAM | %98,5 |
| MolmoAct2 | %98,25 |
| Cosmos 3 Edge | %97,95 |
| π0.5 | %97,5 |
| EVO1 | %96,65 |
| GR00T N1.7 | %96,5 |
| VLA-JEPA | %96,5 |
| FastWAM | %96,4 |
| Multitask DiT | %90,6 |
| π0-FAST | %82,5 |

Bu tabloyu tek bir şey için oku: benchmark doymuş. On model birbirinden en fazla üç puan uzakta duruyor;
her biri yüzlerce gösterimli simüle görevlerde, kendi yazarları tarafından, birbirinden biraz farklı protokollerle
değerlendirilmiş. Hangisinin senin masandaki 3. cebi bulacağını söyleyemez. Senin 24 denemen söyleyebilir.

## Her birini eğitmenin maliyeti

LeRobot donanım rehberinden (batch 8'de tepe bellek):

| Grup | Policy'ler | GPU belleği |
| --- | --- | --- |
| Hafif | ACT | 2 ila 6 GB |
| Diffusion | Multitask DiT | 8 ila 14 GB |
| Küçük VLA | SmolVLA | 10 ila 16 GB |
| Büyük VLA | π0, π0-FAST, π0.5, X-VLA, WALL-OSS, GR00T, EO-1 | 24 ila 40 GB; 24 GB batch 1'de bile dar |
| World-action | Cosmos 3 Edge | LoRA ile 21 GB, tam eğitimde 47 GB |
| | Praxis-1 | LoRA ile 62 GB |

Rehberin kendi pratik kuralı: eğitim, verinin üzerinden 5 ila 10 geçişte yakınsıyor.

## Hangileri bu kolu zaten gördü

Tam olarak senin robotun için bir prior, parametrelerden daha değerli. Şunlarda var:

| Policy | Dokümanlardaki kanıt | Verimize olduğu gibi uyuyor mu? |
| --- | --- | --- |
| SmolVLA | Topluluk SO-100 verisiyle önceden eğitilmiş | Evet: bu rehber |
| GR00T N1.7 | Dokümanlardaki `new_embodiment` örneği SO-101 | Evet |
| G0.5 | `lerobot/g05_so101` checkpoint'i | Action'lar için evet; akıl yürütme alt görev etiketleri ister |
| MolmoAct2 | `lerobot/MolmoAct2-SO100_101-LeRobot` checkpoint'i | Evet, kamera adlarını değiştirerek |
| FLUX 3 Action | `black-forest-labs/flux-3-action-so101` checkpoint'i, 30 Hz, altı eklem, sahne + bilek | Evet, kamera adlarını değiştirerek |
| NAT VLA | Bir SO-100'de rollout örneği; 24 GB'lık bir eğitim tarifi | Evet |

Ve uğraşmadan uymayan iki tane: LingBot-VA ve Praxis-1 eklem açıları değil, end-effector pozları bekliyor.

## Bu proje için kısa liste

| Sıra | Policy | Cevapladığı soru |
| --- | --- | --- |
| 1 | ACT, tek cümle | VLA, küçük bir modelin yapamadığı bir şey yapıyor mu? |
| 2 | GR00T N1.7 | Göreli action'lı 3 B'lik bir VLA cep numarasını daha iyi izliyor mu? |
| 3 | Etiketlenmiş alt görevlerle G0.5 | Önce alt görevi söylemek işe yarıyor mu, ve nerede başarısız olduğunu okuyabiliyor muyum? |
| 4 | FLUX 3 Action | Model kolu zaten tanıyorken video tahmini işe yarıyor mu? |
| 5 | Cosmos 3 Edge | Aynısı, SO-101 prior'ı olmadan ama görülebilen bir tahmin videosuyla |

Hepsini aynı 24 eşleştirilmiş denemeden geçir ([deney planı](next.md)). Teker teker.

## Masaya ne götürmeli

- Bir model ailesi, *başka neyi tahmin edeceğine* dair bir seçimdir: hiçbir şey, metin ya da gelecek.
- Benchmark'lar modelleri başkasının görevinde sıralar. Onları kendi görevinde kendin sırala.
- Varsa, her zaman bir SO-100 ya da SO-101 görmüş bir checkpoint'ten başla.

**Kaynaklar:** her policy için LeRobot dokümantasyon sayfaları,
[compute hardware guide](https://huggingface.co/docs/lerobot/hardware_guide),
[remote inference](https://huggingface.co/docs/lerobot/remote_inference) ve
[annotation pipeline](https://huggingface.co/docs/lerobot/annotation_pipeline). Boyutlar ve sonuçlar Kasım 2026'da
orada yazdığı gibi.
