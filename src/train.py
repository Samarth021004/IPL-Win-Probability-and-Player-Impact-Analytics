"""Train, calibrate and evaluate the win-probability (classification) and projected-score (regression) models.

Honest evaluation: chronological split (train <=2022, validation 2023, test >=2024).
For descriptive WPA over the whole history we also produce out-of-fold predictions
(GroupKFold by match), so no ball is ever scored by a model that saw its own match.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import engineer  # noqa: E402
from sklearn.preprocessing import StandardScaler

CORE = ["runs_needed", "balls_left", "wickets_lost", "rrr"]
MOMENTUM = ["crr", "rrr_minus_crr", "runs_l12", "wkts_l12"]
CONTEXT = ["target", "venue_avg1", "venue_chase_rate", "elo_diff"]
FEATURES = CORE + MOMENTUM + CONTEXT
FEATURES_LR = FEATURES + ["log_rrr", "sqrt_bl", "wk_x_bl"]
MONO = {"runs_needed": -1, "balls_left": 1, "wickets_lost": -1, "rrr": -1}  # more required / fewer resources => lower P(win)

F1_FEATURES = ["cum_runs", "balls_left", "wickets_lost", "crr", "runs_l12", "wkts_l12", "venue_avg1", "bat_adv"]
SEED = 7
LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, n_jobs=2, random_state=SEED, verbose=-1,
)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(logloss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))


def fit_lgb(X, y, Xv, yv, mono_cols, n_estimators=3000):
    cons = [MONO.get(c, 0) for c in mono_cols]
    m = lgb.LGBMClassifier(n_estimators=n_estimators, monotone_constraints=cons, **LGB_PARAMS)
    m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main(proc="data/processed", models="models", reports="reports"):
    proc, models, reports = Path(proc), Path(models), Path(reports)
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    out = {}

    # ===================== classification: P(chasing side wins) =====================
    c = engineer(pd.read_parquet(proc / "chase_states.parquet"))
    tr_all = c[(c.is_start == 0) & (c.terminal == 0)].copy().reset_index(drop=True)
    tr_all["y"] = tr_all.chase_won
    train, val, test = (tr_all[tr_all.year <= 2022], tr_all[tr_all.year == 2023], tr_all[tr_all.year >= 2024])
    out["split"] = {k: dict(rows=int(len(d)), matches=int(d.match_id.nunique()))
                    for k, d in dict(train=train, val=val, test=test).items()}
    print(out["split"])

    def logreg():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))

    results = {}
    # --- candidate models, all fit on <=2022 and scored on the untouched 2024+ test years ---
    lr_core = logreg().fit(train[CORE], train.y)
    results["logistic_core_state"] = metrics(test.y, lr_core.predict_proba(test[CORE])[:, 1])
    lr_base = logreg().fit(train[FEATURES], train.y)
    results["logistic_all_features"] = metrics(test.y, lr_base.predict_proba(test[FEATURES])[:, 1])
    lr = logreg().fit(train[FEATURES_LR], train.y)
    p_test = lr.predict_proba(test[FEATURES_LR])[:, 1]
    results["logistic_engineered (MAIN)"] = metrics(test.y, p_test)

    gbm = fit_lgb(train[FEATURES], train.y, val[FEATURES], val.y, FEATURES)
    results["lightgbm_monotone"] = metrics(test.y, gbm.predict_proba(test[FEATURES])[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(gbm.predict_proba(val[FEATURES])[:, 1], val.y)
    results["lightgbm_monotone+isotonic(2023 val)"] = metrics(test.y, iso.predict(gbm.predict_proba(test[FEATURES])[:, 1]))
    out["test_metrics"] = results
    print(pd.DataFrame(results).T.round(4).to_string())

    # --- grouped 5-fold CV over ALL seasons (1,195 matches): a much larger sample than the 3 test years ---
    folds = list(GroupKFold(n_splits=5).split(tr_all, groups=tr_all.match_id))
    oof_lr = np.zeros(len(tr_all))
    oof_gbm = np.zeros(len(tr_all))
    small_gbm = dict(LGB_PARAMS, num_leaves=4, min_child_samples=1000, learning_rate=0.03, reg_lambda=20)
    for tri, vai in folds:
        oof_lr[vai] = logreg().fit(tr_all.loc[tri, FEATURES_LR], tr_all.loc[tri, "y"]).predict_proba(
            tr_all.loc[vai, FEATURES_LR])[:, 1]
        g = lgb.LGBMClassifier(n_estimators=400, monotone_constraints=[MONO.get(f, 0) for f in FEATURES], **small_gbm)
        g.fit(tr_all.loc[tri, FEATURES], tr_all.loc[tri, "y"])
        oof_gbm[vai] = g.predict_proba(tr_all.loc[vai, FEATURES])[:, 1]
    out["grouped_cv_all_seasons"] = {"logistic_engineered": metrics(tr_all.y, oof_lr),
                                     "lightgbm_regularised": metrics(tr_all.y, oof_gbm)}
    print(json.dumps(out["grouped_cv_all_seasons"], indent=1))

    # calibration curve (chronological test set)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_test, bins) - 1, 0, 9)
    out["calibration"] = [
        dict(bin=f"{bins[i]:.1f}-{bins[i+1]:.1f}", mean_pred=float(p_test[idx == i].mean()),
             frac_win=float(test.y.values[idx == i].mean()), n=int((idx == i).sum()))
        for i in range(10) if (idx == i).sum() > 0
    ]
    # by stage of chase: early predictions should be uncertain, late ones sharp
    stage = pd.cut(120 - test.balls_left, [-1, 36, 72, 102, 120], labels=["overs 1-6", "7-12", "13-17", "18-20"])
    rows = []
    for s_ in stage.cat.categories:
        mask = (stage == s_).values
        rows.append(dict(stage=str(s_), n=int(mask.sum()), auc=float(roc_auc_score(test.y[mask], p_test[mask])),
                         logloss=float(log_loss(test.y[mask], np.clip(p_test[mask], 1e-6, 1 - 1e-6))),
                         mean_abs_conf=float(np.abs(p_test[mask] - 0.5).mean())))
    out["by_stage"] = rows
    print(pd.DataFrame(rows).round(3).to_string())
    # AUC by season (OOF) - shows that some eras are simply more predictable than others
    tmp = tr_all.assign(p=oof_lr)
    out["oof_by_year"] = {int(y): dict(auc=float(roc_auc_score(g.y, g.p)), logloss=float(log_loss(g.y, g.p.clip(1e-6, 1 - 1e-6))))
                          for y, g in tmp.groupby("year")}

    coef = pd.Series(lr[-1].coef_[0], index=FEATURES_LR)
    out["logistic_coefficients_standardised"] = coef.round(3).sort_values().to_dict()

    # ---------- out-of-fold win probability on ALL rows (start + terminal included) for WPA ----------
    c["win_prob"] = np.nan
    for tri, vai in folds:
        held = set(tr_all.loc[vai, "match_id"])
        mdl = logreg().fit(tr_all.loc[tri, FEATURES_LR], tr_all.loc[tri, "y"])
        mask = c.match_id.isin(held)
        c.loc[mask, "win_prob"] = mdl.predict_proba(c.loc[mask, FEATURES_LR])[:, 1]
    c["win_prob"] = np.where(c.terminal == 1, c.terminal_p, c.win_prob)
    assert c.win_prob.notna().all()
    c[["match_id", "seq", "win_prob"]].to_parquet(proc / "chase_probs.parquet", index=False)

    # ---------- final model (all data) for the interactive what-if page ----------
    final = logreg().fit(tr_all[FEATURES_LR], tr_all.y)
    # Saved as plain numbers (not a pickle) so the app works with ANY scikit-learn version, or none at all.
    sc, clf = final[0], final[-1]
    json.dump(dict(features=FEATURES_LR, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                   coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0])), open(models / "win_prob.json", "w"), indent=1)

    # ===================== regression: projected first-innings score =====================
    f = pd.read_parquet(proc / "first_states.parquet")
    f["bat_adv"] = -f.elo_diff  # batting side's Elo edge (batting side = defending team)
    f["remaining"] = f.first_total - f.cum_runs
    ft = f[f.balls_left > 0]
    ftr, fva, fte = ft[ft.year <= 2022], ft[ft.year == 2023], ft[ft.year >= 2024]
    reg_params = {k: v for k, v in LGB_PARAMS.items() if k != "objective"}
    reg = lgb.LGBMRegressor(n_estimators=3000, objective="regression", **reg_params)
    reg.fit(ftr[F1_FEATURES], ftr.remaining, eval_set=[(fva[F1_FEATURES], fva.remaining)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    from sklearn.linear_model import Ridge
    rg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(ftr[F1_FEATURES], ftr.remaining)
    proj_ridge = fte.cum_runs + rg.predict(fte[F1_FEATURES])
    proj = fte.cum_runs + reg.predict(fte[F1_FEATURES])
    naive = fte.cum_runs + fte.cum_runs / (120 - fte.balls_left).clip(lower=1) * fte.balls_left  # raw run rate x balls left
    st = pd.cut(120 - fte.balls_left, [0, 36, 72, 102, 119], labels=["overs 1-6", "7-12", "13-17", "18-20"])
    rr = []
    for s in st.cat.categories:
        mk_ = (st == s).values
        rr.append(dict(stage=str(s), mae_lgbm=float(mean_absolute_error(fte.first_total[mk_], proj[mk_])),
                       mae_naive_run_rate=float(mean_absolute_error(fte.first_total[mk_], naive[mk_]))))
    out["score_regression"] = dict(
        mae_lgbm=float(mean_absolute_error(fte.first_total, proj)),
        mae_ridge=float(mean_absolute_error(fte.first_total, proj_ridge)),
        mae_naive_run_rate=float(mean_absolute_error(fte.first_total, naive)),
        by_stage=rr, best_iteration=int(reg.best_iteration_))
    print(json.dumps(out["score_regression"], indent=1))

    # out-of-fold projected score (ridge: as accurate as LightGBM here, and simpler) for the Match Story chart
    f = f.reset_index(drop=True)
    f["proj_score"] = np.nan
    for tri, vai in GroupKFold(n_splits=5).split(f, groups=f.match_id):
        mk = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(f.loc[tri, F1_FEATURES], f.loc[tri, "remaining"])
        f.loc[vai, "proj_score"] = f.loc[vai, "cum_runs"] + mk.predict(f.loc[vai, F1_FEATURES])
    f[["match_id", "seq", "proj_score"]].to_parquet(proc / "first_proj.parquet", index=False)

    json.dump(out, open(reports / "metrics.json", "w"), indent=1)
    print("saved ->", reports / "metrics.json")


if __name__ == "__main__":
    main()
