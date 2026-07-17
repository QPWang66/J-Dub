# Atomic-action detection: literature thresholds and what jdub adopts

Deep-research pass (2026-07-17): 5 search angles, 18 sources fetched, 89 claims
extracted, top 25 adversarially verified (3 independent refutation votes each;
11 confirmed, 0 refuted, 14 unverified due to verifier infrastructure errors —
marked below). Full trace: workflow `wf_f0387cdf-07a`.

## Verified findings (checked 2-0 or 3-0 against primary PDFs)

### Ball screen (pick-and-roll) — NETS rule, same public 2015-16 SportVU dump we use

Single-frame triangle rule: fire when **handler–screener δa ≤ 6 ft**,
**on-ball-defender–handler δd1 ≤ 6 ft**, **on-ball-defender–screener δd2 ≤ 3 ft**.
Ball-handler requires possession ≥ 5 consecutive frames. Matchup from linear sum
assignment. Manual inspection: **82% precision** (164/200; independent 300-play
set ~81%). Rule-only F1 vs 900 hand-labeled plays: **0.869** (P&R), learned
ceiling (transformer) 0.951. On 632 games the rules yield 45,802 P&R (~72/game).

> Hauri & Vucetic, *Group Activity Recognition in Basketball Tracking Data —
> Neural Embeddings in Team Sports (NETS)*, ECAI 2023.
> https://arxiv.org/abs/2209.00451 — verified 3-0.

### Dribble handoff — NETS

Possession changes between two offensive players who are **< 6.5 ft** apart
(≈ average NBA wingspan). **90.5% precision** (181/200), rule-only F1 0.874.
Main FP: confusion with P&R happening within a few frames.
> Same source, supplementary A — verified 3-0.

### Defender matchup assignment

- Option A (NETS): linear-sum-assignment minimizing summed Euclidean
  defender→attacker distance. Sufficient for the triangle rule. Verified 3-0.
- Option B (better under screens/help): Franks et al. — defender's canonical
  spot is the convex combination **0.62·offender + 0.11·ball + 0.27·hoop**
  (league fit ±0.02, stable with 30 possessions); score matchups by distance to
  that centroid, matchup evolution as an HMM.
> Franks, Miller, Bornn, Goldsberry, *Counterpoints* (SSAC 2015) and AoAS 2015
> (https://arxiv.org/abs/1405.0231) — verified 3-0. The fitted gammas appear
> only in the AoAS paper, not the SSAC version.

### Coverage classification context (M3)

McIntyre/Brooks/Guttag/Wiens (SSAC 2016) contains **no detection geometry**
(detection is delegated to McQueen/Wiens/Guttag SSAC 2014, which remains
unfetched — open question). Reusable pieces, verified 2-0:
- **screen moment** = frame of minimum screener↔on-ball-defender distance;
  features from 10 frames (0.4 s) before it until next shot/pass/turnover.
- 4-class coverage (over/under/switch/trap) via multinomial logistic regression
  on 340 hand-labeled screens: overall accuracy **0.69**; per-class recall
  over 0.83 / under 0.52 / switch 0.62 / trap 0.19. Low-confidence predictions
  are dropped by softmax threshold (h) into an "unclear" class.
> https://www.sloansportsconference.com/research-papers/recognizing-and-analyzing-ball-screen-defense-in-the-nba

### Possession (medium confidence — corroborated inside verified quotes)

Holder = closest player to ball for ≥ 5 consecutive frames, ball within
**5 ft**, below **10 ft height**, slower than 25 ft/s.

## Unverified (verifier infra errors; spot-check PDFs before further tuning)

- Drive (PMC9904462): carrier speed > **5.23 ft/s**, distance-proportion
  (basket-distance reduction ÷ path length) > **0.50**, start distance
  **8.5–28.4 ft** (excludes post moves).
- Cut (same): off-ball speed ≥ **5.96 ft/s**, DP > **0.77**, start ≤ 23.4 ft,
  **end inside 8.5 ft** of the basket. Thresholds from the authors' percentile
  analysis of their own annotations.
- Keshri et al. JQAS 2019: HMM alternative (no hard thresholds); screen
  accuracy 0.868, precision ~0.78 — FP dominated by incidental proximity.
- ETSU thesis (dc.etsu.edu/etd/3908): DHO candidate rule ~90% recall then SVM;
  cites McQueen/Guttag recall 82% / precision 80%.

## What jdub adopts (events.py)

| Component | Old (v1 guess) | New (literature) | Source tier |
|---|---|---|---|
| Holder radius | 3.0 ft | 5.0 ft + ball z < 10 ft, ≥5 frames | medium (NETS) |
| Matchup cost | Σ dist(def, att) | Σ dist(def, 0.62·att+0.11·ball+0.27·hoop) | verified (Franks) |
| Screen rule | screener slow ≤2 ft/s, ≤6 ft of def, ≤12 ft of handler, ≥8 frames | NETS triangle δa≤6, δd1≤6, δd2≤3, ≥3 frames; set-screen fraction feeds confidence | verified (NETS) |
| Handoff distance | 5 ft | 6.5 ft | verified (NETS) |
| Drive | start 15–30 ft, decline ≥8 ft | start 8.5–28.4 ft, DP>0.5, speed>5.23 ft/s | unverified (PMC) |
| Cut | decline ≥10 ft in 1 s, start ≤35 ft | speed≥5.96, DP>0.77, start ≤23.4, end ≤8.5 ft | unverified (PMC) |

Benchmarks to beat: screen precision ≥ 0.82 (NETS floor), F1 ~0.87;
coverage (M3) accuracy ≥ 0.69 (McIntyre floor).

## Open questions

1. McQueen/Wiens/Guttag 2014 exact thresholds (the paper everyone defers to).
2. Do the PMC drive/cut numbers hold against the actual PDF, on NBA 25 Hz data?
3. Matchup recompute cadence (per-frame vs window) — no published ablation.
4. Explicit transition filter — no verified source states one; jdub keeps its
   half-court start-distance caps as a proxy.
