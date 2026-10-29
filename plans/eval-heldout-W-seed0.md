# Held-out eval plan (eval-heldout-W-seed0)

Work order: **white** variant. 6 trays, 32 episodes; each part goes once into the correct bin and once deliberately into the opposite bin (`probe` column).

There is one tray; a tray means laying 6 parts in it and picking them in order. Episode numbers run across the whole plan. Pocket 1 is at the left of the row nearest the arm (seen from behind the arm), 4 at the left of the far row.

## Tray 1

| Pocket | Part | Type |
| --- | --- | --- |
| 1 | `L-W-09` | bracket |
| 2 | `J-T-08` | rim |
| 3 | `B-T-08` | pad |
| 4 | `D-W-09` | gear |
| 5 | `B-W-10` | pad |
| 6 | `D-W-10` | gear |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 1 | 4 | `D-W-09` | put the part in the OK bin |  |
| 2 | 6 | `D-W-10` | put the part in the REJECT bin |  |
| 3 | 5 | `B-W-10` | put the part in the OK bin | yes |
| 4 | 2 | `J-T-08` | put the part in the OK bin | yes |
| 5 | 1 | `L-W-09` | put the part in the REJECT bin | yes |
| 6 | 3 | `B-T-08` | put the part in the OK bin | yes |

## Tray 2

| Pocket | Part | Type |
| --- | --- | --- |
| 1 | `B-T-09` | pad |
| 2 | `B-W-09` | pad |
| 3 | `J-W-09` | rim |
| 4 | `D-T-08` | gear |
| 5 | `J-T-09` | rim |
| 6 | `J-W-10` | rim |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 7 | 6 | `J-W-10` | put the part in the OK bin | yes |
| 8 | 4 | `D-T-08` | put the part in the REJECT bin |  |
| 9 | 2 | `B-W-09` | put the part in the OK bin |  |
| 10 | 3 | `J-W-09` | put the part in the REJECT bin | yes |
| 11 | 1 | `B-T-09` | put the part in the OK bin | yes |
| 12 | 5 | `J-T-09` | put the part in the REJECT bin |  |

## Tray 3

| Pocket | Part | Type |
| --- | --- | --- |
| 2 | `L-W-10` | bracket |
| 3 | `L-T-09` | bracket |
| 4 | `D-T-09` | gear |
| 6 | `L-T-08` | bracket |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 13 | 4 | `D-T-09` | put the part in the REJECT bin |  |
| 14 | 6 | `L-T-08` | put the part in the OK bin | yes |
| 15 | 3 | `L-T-09` | put the part in the OK bin | yes |
| 16 | 2 | `L-W-10` | put the part in the OK bin | yes |

## Tray 4

| Pocket | Part | Type |
| --- | --- | --- |
| 1 | `J-W-09` | rim |
| 2 | `D-W-09` | gear |
| 3 | `B-T-09` | pad |
| 4 | `L-T-08` | bracket |
| 5 | `D-T-09` | gear |
| 6 | `B-W-10` | pad |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 17 | 2 | `D-W-09` | put the part in the REJECT bin | yes |
| 18 | 4 | `L-T-08` | put the part in the REJECT bin |  |
| 19 | 3 | `B-T-09` | put the part in the REJECT bin |  |
| 20 | 6 | `B-W-10` | put the part in the REJECT bin |  |
| 21 | 5 | `D-T-09` | put the part in the OK bin | yes |
| 22 | 1 | `J-W-09` | put the part in the OK bin |  |

## Tray 5

| Pocket | Part | Type |
| --- | --- | --- |
| 1 | `L-W-10` | bracket |
| 2 | `L-W-09` | bracket |
| 3 | `D-T-08` | gear |
| 4 | `J-T-08` | rim |
| 5 | `L-T-09` | bracket |
| 6 | `B-T-08` | pad |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 23 | 1 | `L-W-10` | put the part in the REJECT bin |  |
| 24 | 3 | `D-T-08` | put the part in the OK bin | yes |
| 25 | 4 | `J-T-08` | put the part in the REJECT bin |  |
| 26 | 6 | `B-T-08` | put the part in the REJECT bin |  |
| 27 | 5 | `L-T-09` | put the part in the REJECT bin |  |
| 28 | 2 | `L-W-09` | put the part in the OK bin |  |

## Tray 6

| Pocket | Part | Type |
| --- | --- | --- |
| 1 | `J-W-10` | rim |
| 3 | `D-W-10` | gear |
| 5 | `B-W-09` | pad |
| 6 | `J-T-09` | rim |

| Episode | Pocket | Part | Command | Probe |
| --- | --- | --- | --- | --- |
| 29 | 3 | `D-W-10` | put the part in the OK bin | yes |
| 30 | 5 | `B-W-09` | put the part in the REJECT bin | yes |
| 31 | 1 | `J-W-10` | put the part in the REJECT bin |  |
| 32 | 6 | `J-T-09` | put the part in the OK bin | yes |
