# Sıfırdan flow matching

SmolVLA, π0, GR00T ve Cosmos 3 action'ları aynı şekilde yazıyor: rastgele gürültüden başlıyor ve onu adım adım, bir
harekete dönüşene kadar taşıyorlar. Bu sayfa o fikri sıfırdan kuruyor, aklında tutabileceğin kadar küçük bir problemde
çalıştırıyor, sonra her sembolü [7. adımda](train.md) eğittiğin kodla eşliyor.

## Çözdüğü sorun

Tek durum, iki doğru action: soldan (−1) da sağdan (+1) da kavrayabileceğin bir dişli.

Bir regresyon head'i, insanın yaptığına olan kare hatayı en aza indirecek şekilde eğitilir. Kare hatayı en aza indiren
değer *ortalamadır*. −1 ile +1'in ortalaması 0: dişlinin ortası, kimsenin göstermediği bir kavrama.

Bizim istediğimiz ise bazen −1, bazen +1 veren ve ortayı asla vermeyen bir model. Başka bir deyişle: gösterilen
action'ların dağılımını tek bir sayıyla özetlemek yerine o dağılımdan *örnekleyen* bir model.

## Fikir: cevabı değil, yönü öğren

Gerçek bir action `A` ve bir tutam rastgele gürültü `ε` al. Aralarına düz bir çizgi çek:

```
x_t = t·ε + (1 − t)·A          t = 0 is the action, t = 1 is pure noise
```

Bu çizgi boyunca gidiş yönü sabit: `u = ε − A`. Bir `v(x_t, t)` ağını, rastgele `(A, ε, t)` üçlüleri için bu yönü
verecek şekilde eğit:

```
loss = mean( (v(x_t, t) − (ε − A))² )
```

Üretmek için gürültüden başla ve çizgiyi küçük adımlarla geriye doğru yürü:

```
x = ε
for t in 1.0, 0.9, … 0.1:
    x = x − 0.1 · v(x, t)
```

Yöntemin tamamı bu. Bu da hâlâ bir regresyon; peki neden iki kavramanın ortalamasını almıyor?

## Ortalama neden kayboluyor

Ağ, gürültülü bir `x_t` noktasının hangi `A`'dan geldiğini bilemez, bu yüzden *bu noktayı üretmiş olabilecek bütün
action'lar üzerinden ortalama yönü* öğrenir. "Üretmiş olabilecek" sözünün anlamı yol boyunca değişir:

| Nerede | Ağ ne görüyor | Ne veriyor |
| --- | --- | --- |
| t ≈ 1 (saf gürültü) | İki kavrama da eşit olasılıkta | Ortaya doğru bir yön, ama sadece küçük bir adım atılıyor |
| t ≈ 0,5 | Merkezin solundaki bir nokta neredeyse kesin −1'den gelmiş | −1'e doğru bir yön |
| t ≈ 0 | Nokta kavramalardan birinin hemen yanında | Onun üstüne oturtan ince bir düzeltme |

Yönlerin ortalaması *her noktada ayrı ayrı* alınır ve soldaki noktalara sol kavrama hâkimdir. Merkezin biraz solundan
başlayan bir örnek her adımda daha da sola itilir ve kararını verir. Hangi kavramayı alacağına rastgele başlangıç
noktası `ε` karar verir; onun gerçek bir kavrama olmasına ağ karar verir.

## Çalıştır

`scripts/toy_flow.py` bu sayfanın 50 satırlık hâli: tek durum, −1 ve +1'de gösterimler, aynı küçücük ağla bir
regresyon head'i ve bir flow matching head'i.

```bash
uv run --extra robot python scripts/toy_flow.py
```

```
regression predicts      -0.01   (the average: closes on nothing)
flow matching, 1000 runs 51% near -1, 49% near +1
  mean |x| = 0.98, samples between -0.5 and 0.5: 2.1%
```

Ortada biten %2, on kaba adımın ve küçük bir ağın bedeli: bunlar neredeyse tam sınır çizgisinin üstünden başladı. Daha
fazla adım ya da daha iyi bir ağ bu sayıyı küçültür; sıfıra indirmez. Robotta bu, ara sıra görülen tereddütlü kavramadır
ve parça başına *tek* bir yol göstermen için bir neden daha.

## Oyuncak problemden SmolVLA'ya

Boyutlar dışında hiçbir şey değişmiyor.

| Oyuncak | SmolVLA (`policies/smolvla/modeling_smolvla.py`) |
| --- | --- |
| `A`: tek bir sayı | Gelecekteki 50 action × 6 eklem, normalize edilmiş, 50 × 32'ye sıfırla doldurulmuş |
| `v(x_t, t)` | Action expert. Sahneyi de görüyor: iki görüntüden, cümleden ve eklem durumundan gelen, önbelleğe alınmış 177 token |
| `t`, (0, 1) aralığında düzgün dağılımlı | `t` bir Beta(1.5, 1) dağılımından çekilir, (0.001, 1] aralığına ölçeklenir: gürültülü uçta daha fazla eğitim |
| 10 Euler adımı | 10 Euler adımı (`num_steps`) |
| Durum hiç değişmiyor | Sahne token'ları karar başına bir kez hesaplanır ve on adımın hepsinde yeniden kullanılır |

Neden gürültülü uçta daha fazla eğitim? Görüntülerde zor olan kısım ince ayrıntıdır, o da t = 0 yakınında yaşar.
Action'larda durum tam tersi: chunk büyük oranda denoise edildikten sonra gerisini sahne belirler. Asıl önemli karar
(hangi kavrama, hangi kasa) erkenden, gürültünün içinde verilir.

## Diffusion'dan farkı ne

Diffusion policy'leri (ve onlardan önce görüntü üreticileri) aynı türden bir denoiser öğrenir; pratikte iki fark var.

| | Diffusion (DDPM) | Flow matching |
| --- | --- | --- |
| Veriden gürültüye giden yol | Çok sayıda küçük gürültüleme adımından oluşan eğri bir program | Düz bir çizgi |
| Ağ neyi tahmin ediyor | Eklenen gürültüyü | Çizgi boyunca hızı |
| Inference'ta adım sayısı | Genelde 50 ila 100 (DDIM sampler ile 10 civarı) | Düz Euler adımlarıyla yaklaşık 10 |

On adımın yetmesi düz yollar sayesinde; bir VLA'nın bir chunk'ı saniyenin küçük bir kesrinde üretebilmesi de on adım
sayesinde.

## Karşına çıkacak üç ayar

- **Adım sayısı.** Daha az adım daha hızlı ve daha kaba demek. SmolVLA 10 kullanıyor. Cosmos 3 daha iyi bir solver
  (UniPC) kullanıyor: LIBERO tarifinde 30 adım, DROID tarifinde 4.
- **Guidance.** Ağı iki kez çalıştır, cümleyle ve cümlesiz, ve sonucu koşulsuz olandan uzağa it. Modelin talimatı daha
  sıkı izlemesini sağlar, maliyeti iki katına çıkar. SmolVLA bunu kullanmıyor; Cosmos 3 DROID tarifi kullanıyor
  (ölçek 3).
- **Başlangıç gürültüsü.** Aynı gözlem, farklı `ε`, farklı chunk. Birebir aynı başlangıçtan iki rollout'un farklı
  çıkabilmesi ve [real-time chunking](learn-vla.md)'in yeni chunk'ı eskisine doğru çekmek zorunda olması bu yüzden.

## Masaya ne götürmeli

- Düşük bir flow matching loss'u, *yönün* senin verinde iyi tahmin edildiği anlamına gelir. Koldaki başarı hakkında
  yine hiçbir şey söylemez; bkz. [bir policy'yi ölçmek](learn-eval.md).
- İki gösterim birbiriyle çelişiyorsa model ikisini de yeniden üretir, ara sıra da ikisinin arasında bir şey.
- Örnekleme rastgeledir. Bir policy'yi tek denemeyle değil, çok sayıda denemeyle değerlendir.

**Kaynaklar:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) ·
[Rectified Flow](https://arxiv.org/abs/2209.03003) · [π0](https://arxiv.org/abs/2410.24164) (Beta zaman
örneklemesi) · [Diffusion Policy](https://arxiv.org/abs/2303.04137) · LeRobot 0.6.1 kaynak kodu ·
[LeRobot Cosmos 3 docs](https://huggingface.co/docs/lerobot/cosmos3)
