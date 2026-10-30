<p align="center"><a href="README.md">English</a> · Türkçe</p>

<p align="center">
  <img src="docs/media/final.png" width="900" alt="MuJoCo'da SO-101 kalite kontrol hücresi: genel görünüm, çalışma alanı, yandan görünüm, kamera başlığı, üst kamera ve bilek kamerası">
</p>

<h1 align="center">so101-flywheel</h1>

<p align="center"><b>SO-101 kolu için kendini geliştiren bir veri çarkı, hat sonu kalite kontrol görevi üzerinde.</b><br>
Kaydet, doğrula, düzelt, yeniden eğit. Simülasyon MuJoCo'da, baskılar Bambu X2D'de, eğitim <a href="https://github.com/huggingface/lerobot">LeRobot</a> ile.</p>

<p align="center">
  <a href="https://github.com/TheRobotStudio/SO-ARM100"><img src="https://img.shields.io/badge/robot-SO--101-ff7a1a?labelColor=1c1c1c" alt="SO-101"></a>
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-0.6.1-ff7a1a?labelColor=1c1c1c" alt="LeRobot 0.6.1"></a>
  <a href="https://github.com/google-deepmind/mujoco"><img src="https://img.shields.io/badge/MuJoCo-3.12-ededed?labelColor=1c1c1c" alt="MuJoCo 3.12"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12-ededed?labelColor=1c1c1c" alt="Python 3.12"></a>
  <a href="https://flywheel.getporti.com/?lang=tr"><img src="https://img.shields.io/badge/ilerleme-canl%C4%B1-ff7a1a?labelColor=1c1c1c" alt="Canlı ilerleme sitesi"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/lisans-MIT-ededed?labelColor=1c1c1c" alt="MIT lisansı"></a>
</p>

<p align="center">
  <a href="docs/PRD.md">PRD</a> ·
  <a href="docs/experiment-design.md">Deney tasarımı</a> ·
  <a href="docs/sim.md">Simülasyon</a> ·
  <a href="hardware/README.md">Baskı dosyaları</a> ·
  <a href="docs/report/so101-flywheel-architecture.md">Mimari</a> ·
  <a href="https://flywheel.getporti.com/?lang=tr">İlerleme</a>
</p>

<p align="center"><sub>Dokümanlar İngilizce.</sub></p>

## Ne

Kol bir görevi her denemede kaydeder, bir VLM verifier başarıyı değerlendirir, başarısız denemeler insan düzeltmesiyle
yeni veriye dönüşür ve policy retrain → eval → deploy döngüsünden geçer. İlk referans görev otomotiv yan sanayi hat
sonu kalite kontrolü: kol jant, dişli, balata ve braketleri kameraya gösterir; iş emrine uyan sağlam parçalar OK
kasasına, kusurlular ve yanlış varyantlar ret kasasına gider.

| | |
| --- | --- |
| 🧪 **Simülasyon** | hücrenin MuJoCo sahnesi, scripted uzman, dengeli kayıt ve held-out eval planları |
| 🖨️ **Donanım** | tek `params.yaml` hem simi hem parametrik CAD'i besler; 18 numaralı baskı tablası |
| 🦾 **Gerçek kol** | leader → follower teleop, sesli yönlendiren plan güdümlü kaydedici, çökme kurtarma |
| 📈 **Eğitim** | kaydedilen dataset üzerinde ACT, Hugging Face Hub'a private yüklenir |

## Durum

Faz 0. Repoda referans görevin MuJoCo sahnesi, onu doğrulayan scripted bölüm, aynı ölçülerden üretilen baskı
dosyaları, teleop ve kaydedici var. Verifier, review UI ve eğitim döngüsü henüz yok.

## Başla

```bash
cd python
uv sync
uv run flywheel-sim view
```

### Gerçek kol: teleop

Robot bağımlılıkları `robot` extra'sında (sim kurulumuna girmez):

```bash
cd python
uv sync --extra robot
uv run flywheel-robot teleop                     # Ctrl+C ile çık
uv run flywheel-robot teleop --top 1 --wrist 0   # kameralarla, Rerun açılır
```

Kalibrasyon `hardware/calibration/{follower,leader}.json` içinde. Yeniden kalibre etmek için
`uv run flywheel-robot calibrate follower|leader|both`; ilk soruya `c` + ENTER. Eski dosya
`hardware/calibration/old/` altına yedeklenir. Portlar komutta sabit; seninkiler farklıysa `--follower-port` /
`--leader-port` ile ver. Başlamadan önce follower'ın 12 V adaptörünü tak ve leader'ı follower'ın pozuna yakın tut:
bağlanınca follower doğrudan leader'ın pozuna gider.

### Gerçek kol: kayıt

Kayıt planı (`plans/record-W-seed0`) üç seviyedir; numaralar terminalde, plan dosyasında ve sitede aynıdır:

| Seviye | Ne | Sayı |
| --- | --- | --- |
| Bölüm | Tek parça: al, kameraya göster, kasaya bırak | 1–120 |
| Tepsi | Tepsiye dizilmiş 6 parça; 6 bölümde tepsi boşalır | 1–20 |
| Seans | Tek `record --session N` çalıştırması: 5 tepsi, 30 bölüm, 30 farklı parça | 1–4 |

Kaydedici sıradaki tepsiyi, cebi ve kasayı ekranda gösterip sesle söyler; operatör → ya da ENTER ile ilerler (başlat →
kameraya gösterdim → bıraktım); ← bölümü yeniden çeker, iki kez Esc çıkar. `lerobot-record`'daki gibi tuşlar globaldir,
Rerun penceresi öndeyken de çalışır. Her karenin görev metni fazına göre yazılır, ACT'nin komutu
`observation.environment_state` kanalına gider. Kol `lerobot-teleoperate` gibi 60 Hz'de sürülür, her ikinci adım
30 fps dataset'e yazılır. Seans bitince `check` çalışır (eklem aralıkları, follower'ın leader'ı izlemesi, komut
kanalının planla eşleşmesi, videolar), ancak ondan sonra dataset private olarak Hugging Face Hub'a yüklenir
(`--no-push` ile atlanır). Sesli yönlendirmeler İngilizcedir.

```bash
cd python
uv run flywheel-robot sessions                    # seansların durumu, sıradakinin parça listesi
uv run flywheel-robot record --session 1          # üst kamera 1, bilek 0; Rerun paneli açılır (--no-display ile kapat)
uv run flywheel-robot record --session 1 --fake   # donanımsız deneme
uv run flywheel-robot check                       # dataset'i elle kontrol et (--repo-id fport/so101-qc-test)
```

Dataset `outputs/datasets/so101-qc` altına yazılır ve her tepsi sonunda `so101-qc.snapshot`'a yedeklenir. Kayıt
temiz kapanmazsa (çökme, elektrik) bir sonraki çalıştırma son tepsi yedeğine döner ve kaldığı yerden sürer.
Hangi bölümün hangi plan satırı olduğu `so101-qc/flywheel/episodes.jsonl`'da.

### Eğitim (ACT v0)

```bash
cd python
uv run lerobot-train \
  --dataset.repo_id=fport/so101-qc --dataset.video_backend=pyav \
  --policy.type=act --policy.device=mps \
  --policy.repo_id=fport/so101-qc-act-v0 --policy.private=true \
  --output_dir=outputs/train/act-v0 --job_name=act-v0 \
  --wandb.enable=false
```

`pyav`: Homebrew'un FFmpeg 9'u torchcodec ile yüklenemiyor. GPU'lu makinede `--policy.device=cuda`.

## Yapı

```
docs/        PRD, baskı planı, deney tasarımı, sim notları, mimari raporu
plans/       dengeli kayıt ve eval planları
hardware/    params.yaml (ortak ölçü kaynağı), print/ (numaralı baskı tablaları), calibration/
python/      flywheel.sim (MuJoCo), flywheel.hardware (parametrik CAD), flywheel.robot (teleop, kayıt)
site/        ilerleme sitesi (flywheel.getporti.com, Cloudflare Workers)
```

## Lisans

MIT, bkz. [LICENSE](LICENSE). `python/flywheel/sim/assets/so101/` altındaki SO-101 modeli TheRobotStudio'nundur ve
Apache-2.0 lisanslıdır ([NOTICE](python/flywheel/sim/assets/so101/NOTICE.md)).
