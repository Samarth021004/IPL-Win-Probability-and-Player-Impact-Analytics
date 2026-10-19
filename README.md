# Pressure Index: who actually delivers when an IPL chase is on the line?

A ball-by-ball win-probability engine for IPL run chases (2008-2026), with a Streamlit app built on top of it.
Every delivery of every second innings gets a win probability; the change in that probability is credited to the
batter and the bowler (Win Probability Added, WPA), weighted by how much the situation mattered (leverage).

## Run it

```bash
pip install -r requirements.txt
# data/raw/ipl_json/ must contain the Cricsheet IPL JSON files (https://cricsheet.org/downloads/ipl_json.zip)
./run_pipeline.sh            # parse -> features -> train -> WPA (about 20 seconds)
streamlit run app/app.py
```

The app itself only needs streamlit, pandas, numpy, plotly and pyarrow: the win-probability model is stored as plain coefficients in `models/win_prob.json`, so it does not depend on your scikit-learn version.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `src/parse.py` | Flattens JSON to one row per delivery. Legal-ball counting, season from match date (Cricsheet's `season` labels are inconsistent), team/venue name standardisation, flags DLS / ties / no-results. |
| 2 | `src/features.py` | State after each ball: runs needed, balls left, wickets, required and current rate, last-12-ball momentum, plus pre-match Elo and venue history computed **only from earlier matches**. |
| 3 | `src/train.py` | Win-probability classifier and first-innings projected-score regressor. Chronological test (train <= 2022, test 2024-26) plus grouped 5-fold CV over all seasons. Writes out-of-fold probabilities. |
| 4 | `src/wpa.py` | Per-ball WPA, leverage index, clutch score, match-level bootstrap intervals, empirical-Bayes shrinkage. |
| 5 | `app/app.py` | Match Story, Player Explorer, Clutch Leaderboard, Chase Simulator, Model Card. |

## Results (all in `reports/metrics.json` and the app's Model Card)

| Model (test = 210 matches, 2024-26) | Log loss | Brier | AUC |
|---|---|---|---|
| **Logistic regression + 3 smooth terms (used)** | **0.401** | **0.130** | **0.911** |
| LightGBM, monotone constraints | 0.421 | 0.133 | 0.909 |
| LightGBM + isotonic calibration (2023 validation) | 0.435 | 0.141 | 0.894 |

Grouped CV over all 1,191 chases: logistic 0.428 vs regularised LightGBM 0.443 log loss.

Things worth knowing, because they are what the data actually showed:

- **Simple beat complex.** ~1,200 matches (120 highly correlated balls each) is too little independent signal for boosted trees. I tried heavier regularisation, splines, polynomial interactions and residual boosting on top of the logistic; none improved on it in grouped CV. The first-innings score regression shows the same pattern (ridge MAE 20.1 runs, LightGBM 20.1, naive run-rate projection 28.7).
- **Early-ball bug found and fixed.** The first version fed the raw current run rate to the model, so one boundary off ball 1 (24 runs an over) swung win probability by ~15 points, against ~3.5 for the same shot mid-innings. That inflated openers' WPA. Run-rate and last-12-ball features are now shrunk towards a typical IPL rate when few balls have been bowled (`smooth_crr` in `src/features.py`); a first-ball four is now worth ~4.6 points, in line with every other stage.
- **Post-hoc calibration hurt.** Isotonic regression fit on the 71-match validation season overfit. Maximum-likelihood logistic probabilities are already well calibrated (the Model Card shows the curve).
- **Test years are an easy era.** Out-of-fold AUC is 0.89-0.92 for 2024-26 but 0.79-0.81 for 2017, 2018 and 2023, so the headline AUC flatters the model relative to the full history.
- **WPA is internally consistent.** Per match it sums to (result - starting win probability) with a residual of ~1e-16, and the largest swings are the famous ones (Dhoni 2016, Tewatia 2022, Rinku Singh 2023, Abdul Samad 2023).
- **Clutch is mostly noise.** For batters the estimated true spread in clutch is indistinguishable from zero, and for bowlers it is only ~16% of the observed spread, so the app shows raw scores with intervals next to shrunk scores and states the reliability up front. Total WPA is the more trustworthy measure.

## Definitions

- **WPA** = win probability after a ball minus before, from out-of-fold predictions. Batter +, bowler -; wides go to the bowler only.
- **Leverage index** = average |WPA| of balls in comparable situations (same win-probability band and phase) / overall average. 1.0 is an average ball.
- **Clutch** = WPA / avg leverage - sum(WPA / leverage). Positive means more value when the situation mattered more.

## Limitations

- Only chases are scored; first-innings contributions earn no WPA.
- Credit goes to the batter on strike and the bowler; fielders, run-outs and partnerships are not separated.
- The model does not know who is playing, so WPA measures value relative to an average ball.
- Excluded: 25 ties/no-results, 23 DLS matches, and 4 chases that ended without a natural conclusion.
- Eras differ (Impact Player rule from 2023, rising scoring rates); Elo and venue history are the only context features.

## Data

Ball-by-ball data from [Cricsheet](https://cricsheet.org), used under the ODC Attribution License (ODC-By). Please keep the credit if you publish this.
