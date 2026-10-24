#!/usr/bin/env python3
"""Repeated student-grouped benchmark for eventual-dropout cascade.

Compares linear/tree/boosted models, group-level random oversampling, and
feature-family ablations. Every preprocessing fit and oversampling operation is
restricted to the inner training fold. Model/threshold selection happens inside
each outer training partition; outer held-out students are never used to select.
"""
from __future__ import annotations
import argparse, json, warnings, itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ["age_group", "country", "device", "signup_month", "signup_weekday"]
COMPACT = ["score_valid_count", "score_missing_or_invalid_fraction", "score_mean", "score_last_valid", "score_last3_mean", "time_valid_count", "time_missing_fraction", "time_mean", "time_last_valid", "time_last3_mean", "time_total_valid_sum", "mean_gap_hours", "last_gap_hours", "history_span_hours"]
TRENDS = ["score_slope", "time_slope"]
MODULE = [f"lesson_L{i}_{suffix}" for i in (1,2,3) for suffix in ("attempt_count", "score_mean", "time_mean")]
FEATURE_SETS = {
    "demographics": [],
    "compact": COMPACT,
    "compact_plus_trends": COMPACT + TRENDS,
    "compact_plus_module": COMPACT + MODULE,
}

# Sampling is implemented at student-group level: oversample positive students,
# retaining their complete available prefix set. Each selected student-copy has
# total sample weight 1, so long-history students do not dominate by row count.
CONFIGS = [
    {"name":"logistic", "family":"logistic", "balance":"none", "sampling":"none"},
    {"name":"logistic_class_weight", "family":"logistic", "balance":"balanced", "sampling":"none"},
    {"name":"logistic_group_oversample", "family":"logistic", "balance":"none", "sampling":"group_oversample"},
    {"name":"gradient_boosting", "family":"gradient_boosting", "balance":"none", "sampling":"none"},
    {"name":"gradient_boosting_group_oversample", "family":"gradient_boosting", "balance":"none", "sampling":"group_oversample"},
]


def feature_frame(df, stage, feature_set):
    x = df.drop(columns=["student_id", "target", "_row_id", "snapshot_index", "target_observed", "target_reason_audit", "eventual_attempt_count_audit", "prefix_length_audit", "next_attempt_available_audit"], errors="ignore").copy()
    if "signup_date" in x:
        d = pd.to_datetime(x.pop("signup_date"), errors="coerce")
        x["signup_month"] = d.dt.month.astype(float)
        x["signup_weekday"] = d.dt.dayofweek.astype(float)
    keep = list(CATS)
    if stage != "A":
        keep += FEATURE_SETS[feature_set]
    x = x[[c for c in keep if c in x.columns]]
    cats = [c for c in CATS if c in x.columns]
    nums = [c for c in x.columns if c not in cats]
    return x, cats, nums


def build_preprocessor(cats, nums, family):
    transformers=[]
    if nums:
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if family == "logistic": steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), nums))
    if cats:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def build_estimator(cfg, seed):
    fam=cfg["family"]
    if fam == "logistic":
        return LogisticRegression(C=0.5, solver="liblinear", max_iter=200, class_weight="balanced" if cfg["balance"]=="balanced" else None, random_state=seed)
    if fam == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=40, learning_rate=0.04, max_depth=2, min_samples_leaf=10, subsample=0.8, random_state=seed)
    raise ValueError(fam)


def sampled_training_rows(X, y, student_ids, sampling, seed):
    rng=np.random.default_rng(seed)
    sid=np.asarray(student_ids).astype(str); y=np.asarray(y).astype(int)
    groups={}
    for idx, student in enumerate(sid): groups.setdefault(student, []).append(idx)
    unique=list(groups)
    # Label must be constant within each student.
    group_label={s:int(y[groups[s][0]]) for s in unique}
    pos=[s for s in unique if group_label[s]==1]; neg=[s for s in unique if group_label[s]==0]
    selected=list(unique)
    if sampling=="group_oversample" and pos and neg:
        selected += list(rng.choice(pos, size=max(0,len(neg)-len(pos)), replace=True))
    # Materialize each selected student-copy and assign each copy total weight 1.
    ix_parts=[]; w_parts=[]
    for s in selected:
        ix=np.asarray(groups[s],dtype=int)
        ix_parts.append(ix)
        w_parts.append(np.full(len(ix),1.0/max(1,len(ix)),dtype=float))
    idx=np.concatenate(ix_parts) if ix_parts else np.arange(len(y))
    weights=np.concatenate(w_parts) if w_parts else np.ones(len(y))
    return X.iloc[idx], y[idx], weights


def fit_config(X, y, student_ids, cats, nums, cfg, seed):
    # Fit preprocessing on the original training fold only, then train estimator
    # on student-group-oversampled transformed rows when requested.
    prep=build_preprocessor(cats,nums,cfg["family"])
    Xt=prep.fit_transform(X)
    Xs, ys, weights=sampled_training_rows(X, y, student_ids, cfg["sampling"], seed)
    # Use the fitted transform from the original fold for sampled rows.
    ix_map = None
    # Since Xs is made of X rows in order-preserving group chunks, transform directly.
    Xst=prep.transform(Xs)
    est=build_estimator(cfg,seed)
    est.fit(Xst,ys,sample_weight=weights)
    return prep,est


def predict_config(fitted, X):
    prep,est=fitted
    return est.predict_proba(prep.transform(X))[:,1]


def make_stage_data(input_dir, audit):
    a=pd.read_csv(input_dir/"A_model_table.csv").drop(columns=["target"],errors="ignore")
    a=a.merge(audit[["student_id","eventual_dropout"]],on="student_id",how="left",validate="one_to_one").rename(columns={"eventual_dropout":"target"})
    a["_row_id"]=np.arange(len(a))
    bparts=[pd.read_csv(input_dir/"B_rolling_early.csv"),pd.read_csv(input_dir/"B_rolling_early_censored_rows.csv")]
    b=pd.concat(bparts,ignore_index=True).drop_duplicates(["student_id","snapshot_index"])
    b=b.drop(columns=["target","target_observed","target_reason_audit","eventual_attempt_count_audit","prefix_length_audit","next_attempt_available_audit"],errors="ignore")
    b=b.merge(audit[["student_id","eventual_dropout"]],on="student_id",how="left",validate="many_to_one").rename(columns={"eventual_dropout":"target"})
    b["_row_id"]=np.arange(len(b))
    cparts=[pd.read_csv(input_dir/"C2_rolling.csv"),pd.read_csv(input_dir/"C2_censored_rows.csv")]
    c=pd.concat(cparts,ignore_index=True).drop_duplicates(["student_id","snapshot_index"])
    c=c.drop(columns=["target","target_observed","target_reason_audit","eventual_attempt_count_audit","prefix_length_audit","next_attempt_available_audit"],errors="ignore")
    c=c.merge(audit[["student_id","eventual_dropout"]],on="student_id",how="left",validate="many_to_one").rename(columns={"eventual_dropout":"target"})
    c["_row_id"]=np.arange(len(c))
    out={"A":a,"B":b,"C":c}
    for stage,df in out.items():
        if df.target.isna().any(): raise ValueError(f"Missing eventual label for stage {stage}")
        df.target=df.target.astype(int); df.student_id=df.student_id.astype(str)
        if df.groupby("student_id").target.nunique().max()>1: raise AssertionError("Label inconsistent within student")
    return out


def student_max_scores(df, row_scores, ids):
    tmp=pd.DataFrame({"student_id":df.student_id.astype(str).to_numpy(),"score":np.asarray(row_scores,float)})
    return tmp.groupby("student_id").score.max().reindex(ids).to_numpy(dtype=float)


def thresholds(scores, n=4):
    x=np.asarray(scores,float); x=x[np.isfinite(x)]
    if len(x)==0: return [float("inf")]
    return sorted(set(np.quantile(x,np.linspace(0,1,n)).tolist()+[float(np.nextafter(x.max(),np.inf))]))


def policy_metrics(y, flags):
    y=np.asarray(y,int); flags=np.asarray(flags,bool)
    tn,fp,fn,tp=confusion_matrix(y,flags.astype(int),labels=[0,1]).ravel()
    return {"precision":float(precision_score(y,flags,zero_division=0)),"recall":float(recall_score(y,flags,zero_division=0)),"f1":float(f1_score(y,flags,zero_division=0)),"flagged":int(flags.sum()),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"n":len(y),"positives":int(y.sum()),"prevalence":float(y.mean())}


def choose_policy(y, score_by_stage, config_scores, precision_floor, top_k=3):
    # Keep the top-k configurations by stage OOF AP, then optimize combinations
    # and thresholds on inner OOF predictions only.
    top={}; ap_rows=[]
    for stage, configs in config_scores.items():
        vals=[]
        for name,scores in configs.items():
            ok=np.isfinite(scores)
            ap=float(average_precision_score(y[ok],scores[ok])) if ok.sum() and len(np.unique(y[ok]))==2 else float("nan")
            ap_rows.append({"stage":stage,"config":name,"oof_ap":ap,"n_scored":int(ok.sum()),"positives_scored":int(y[ok].sum())})
            if np.isfinite(ap): vals.append((ap,name))
        top[stage]=[n for _,n in sorted(vals,reverse=True)[:top_k]]
    best=None
    for combo in itertools.product(top["A"],top["B"],top["C"]):
        scores={s:config_scores[s][cfg] for s,cfg in zip(["A","B","C"],combo)}
        cand={s:thresholds(scores[s]) for s in scores}
        for ta,tb,tc in itertools.product(cand["A"],cand["B"],cand["C"]):
            flags=np.zeros(len(y),dtype=bool)
            for s,t in zip(["A","B","C"],[ta,tb,tc]):
                flags |= np.isfinite(scores[s]) & (scores[s]>=t)
            n_flagged=int(flags.sum())
            if n_flagged==0: continue
            tp=int(np.sum(flags & (np.asarray(y,int)==1)))
            precision=tp/n_flagged
            if precision+1e-12<precision_floor: continue
            recall=tp/max(1,int(np.sum(y)))
            m={"precision":precision,"recall":recall,"flagged":n_flagged,"tp":tp,"fp":n_flagged-tp,"fn":int(np.sum(y))-tp}
            # maximize recall, then precision, then lower alert burden
            key=(recall,precision,-n_flagged)
            if best is None or key>best[0]: best=(key,combo,(ta,tb,tc),m)
    return best, pd.DataFrame(ap_rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="data/cascade_snapshots")
    ap.add_argument("--output-dir",default="outputs/repeated_boosting_imbalance")
    ap.add_argument("--outer-seeds",type=int,nargs="+",default=[42])
    ap.add_argument("--inner-folds",type=int,default=2)
    ap.add_argument("--precision-floors",type=float,nargs="+",default=[.40])
    ap.add_argument("--top-k",type=int,default=3)
    ap.add_argument("--compact-only",action="store_true",help="Restrict B/C to compact features for a faster independent split check")
    ap.add_argument("--feature-sets",nargs="+",choices=["compact","compact_plus_trends","compact_plus_module"],default=["compact","compact_plus_trends","compact_plus_module"])
    args=ap.parse_args(); root=Path.cwd(); inp=Path(args.input_dir); out=Path(args.output_dir)
    if not inp.is_absolute(): inp=root/inp
    if not out.is_absolute(): out=root/out
    out.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(inp/"snapshot_audit.csv")[["student_id","dropout_binary_audit_only"]].drop_duplicates("student_id")
    audit=audit.rename(columns={"dropout_binary_audit_only":"eventual_dropout"}); audit.student_id=audit.student_id.astype(str); audit.eventual_dropout=audit.eventual_dropout.astype(int)
    stages=make_stage_data(inp,audit)
    config_by_name={c["name"]:c for c in CONFIGS}
    config_feature=[]
    for stage in stages:
        fsets=["demographics"] if stage=="A" else (["compact"] if args.compact_only else args.feature_sets)
        for fs in fsets:
            for cfg in CONFIGS: config_feature.append((stage,fs,cfg))
    all_policy=[]; all_cv_ap=[]; all_fold_scores=[]; all_stage_summaries=[]; all_test_predictions=[]
    for rep,seed in enumerate(args.outer_seeds,1):
        print(f"\n=== Outer repeat {rep}/{len(args.outer_seeds)} seed={seed} ===",flush=True)
        meta=audit.copy().sort_values("student_id").reset_index(drop=True)
        splitter=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=seed)
        tr_ix,te_ix=next(splitter.split(meta.student_id.to_numpy().reshape(-1,1),meta.eventual_dropout.to_numpy()))
        dev_meta=meta.iloc[tr_ix].reset_index(drop=True); test_meta=meta.iloc[te_ix].reset_index(drop=True)
        dev_ids=set(dev_meta.student_id); test_ids=set(test_meta.student_id)
        assert not (dev_ids & test_ids)
        y_dev=dev_meta.eventual_dropout.to_numpy(); y_test=test_meta.eventual_dropout.to_numpy(); dev_order=dev_meta.student_id.tolist(); test_order=test_meta.student_id.tolist()
        nfold=min(args.inner_folds,int(y_dev.sum()),int((y_dev==0).sum()))
        cv=StratifiedGroupKFold(n_splits=nfold,shuffle=True,random_state=seed+7)
        fold_list=list(cv.split(dev_meta.student_id.to_numpy().reshape(-1,1),y_dev,groups=dev_meta.student_id.to_numpy()))
        oof_by_stage={}; config_meta={}; feature_info={}
        # Train inner OOF scores for every stage/config/feature set.
        for config_i,(stage,fs,cfg) in enumerate(config_feature,1):
            if config_i % 5 == 0: print(f"  inner OOF config {config_i}/{len(config_feature)}", flush=True)
            df=stages[stage]
            X,cats,nums=feature_frame(df,stage,fs)
            row_sid=df.student_id.astype(str).to_numpy(); row_y=df.target.to_numpy()
            oof_rows=np.full(len(df),np.nan)
            for fold,(ti,vi) in enumerate(fold_list,1):
                tr_ids=set(dev_meta.student_id.iloc[ti]); va_ids=set(dev_meta.student_id.iloc[vi])
                tr=np.isin(row_sid,list(tr_ids)); va=np.isin(row_sid,list(va_ids))
                if not va.any() or not tr.any(): continue
                fitted=fit_config(X.loc[tr].reset_index(drop=True),row_y[tr],row_sid[tr],cats,nums,cfg,seed+fold)
                oof_rows[va]=predict_config(fitted,X.loc[va].reset_index(drop=True))
            student_scores=student_max_scores(df.loc[df.student_id.isin(dev_ids)].reset_index(drop=True),oof_rows[df.student_id.isin(dev_ids).to_numpy()],dev_order)
            key=f"{fs}::{cfg['name']}"
            oof_by_stage.setdefault(stage,{})[key]=student_scores
            config_meta[(stage,key)]=(fs,cfg)
            if stage not in feature_info: feature_info[stage]={}
            feature_info[stage][key]=(X,cats,nums)
        # Stage AP tables, top candidates and threshold-constrained joint policy.
        for floor in args.precision_floors:
            print(f"  threshold selection floor={floor}",flush=True)
            best, apdf=choose_policy(y_dev,oof_by_stage, oof_by_stage, floor, top_k=args.top_k)
            apdf["repeat"]=rep; apdf["seed"]=seed; apdf["precision_floor"]=floor
            all_cv_ap.append(apdf)
            if best is None:
                chosen={s:(max([(average_precision_score(y_dev[np.isfinite(sc)],sc[np.isfinite(sc)]),k) for k,sc in oof_by_stage[s].items() if np.isfinite(sc).sum() and len(np.unique(y_dev[np.isfinite(sc)]))==2])[1]) for s in stages}
                th={s:float("inf") for s in stages}; devmet=policy_metrics(y_dev,np.zeros(len(y_dev),bool)); feasible=False
            else:
                _,combo,th_tuple,devmet=best; chosen=dict(zip(["A","B","C"],combo)); th=dict(zip(["A","B","C"],th_tuple)); feasible=True
            # Fit selected config on full outer development and score test checkpoints.
            test_scores={}
            for stage in stages:
                key=chosen[stage]; fs,cfg=config_meta[(stage,key)]; df=stages[stage]
                X,cats,nums=feature_frame(df,stage,fs); row_sid=df.student_id.astype(str).to_numpy(); row_y=df.target.to_numpy()
                tr=np.isin(row_sid,list(dev_ids)); te=np.isin(row_sid,list(test_ids))
                fitted=fit_config(X.loc[tr].reset_index(drop=True),row_y[tr],row_sid[tr],cats,nums,cfg,seed+999)
                row_prob=predict_config(fitted,X.loc[te].reset_index(drop=True))
                test_scores[stage]=student_max_scores(df.loc[te].reset_index(drop=True),row_prob,test_order)
                all_stage_summaries.append({"repeat":rep,"seed":seed,"precision_floor":floor,"stage":stage,"config":key,"feature_set":fs,"model":cfg["name"],"sampling":cfg["sampling"],"balance":cfg["balance"]})
            flags=np.zeros(len(test_meta),dtype=bool)
            for stage in stages: flags |= np.isfinite(test_scores[stage]) & (test_scores[stage]>=th[stage])
            m=policy_metrics(y_test,flags)
            all_policy.append({"repeat":rep,"seed":seed,"precision_floor":floor,"development_feasible":feasible,"selected_A":chosen["A"],"selected_B":chosen["B"],"selected_C":chosen["C"],"threshold_A":th["A"],"threshold_B":th["B"],"threshold_C":th["C"],**{f"development_{k}":v for k,v in devmet.items()},**{f"test_{k}":v for k,v in m.items()},"precision_floor_met_on_test":bool(m["flagged"]>0 and m["precision"]+1e-12>=floor)})
            for i,sid in enumerate(test_order):
                all_test_predictions.append({"repeat":rep,"seed":seed,"precision_floor":floor,"student_id":sid,"target":int(y_test[i]),"flag":int(flags[i]),"score_A":test_scores["A"][i],"score_B":test_scores["B"][i],"score_C":test_scores["C"][i]})
        print(f"repeat {rep} done",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    policy=pd.DataFrame(all_policy); apdf=pd.concat(all_cv_ap,ignore_index=True); stage_sel=pd.DataFrame(all_stage_summaries); preds=pd.DataFrame(all_test_predictions)
    policy.to_csv(out/"repeated_outer_cascade_metrics.csv",index=False)
    apdf.to_csv(out/"inner_oof_config_ap.csv",index=False)
    stage_sel.to_csv(out/"selected_stage_configs_by_repeat.csv",index=False)
    preds.to_csv(out/"outer_test_student_predictions.csv",index=False)
    # Summarize across independent outer split runs; report mean/std and precision-floor pass rate.
    metric_cols=["test_precision","test_recall","test_flagged","test_tp","test_fp","test_fn"]
    summary=policy.groupby("precision_floor").agg({**{c:["mean","std","min","max"] for c in metric_cols},"precision_floor_met_on_test":"mean","development_feasible":"mean"})
    summary.columns=["_".join([x for x in col if x]) for col in summary.columns]
    summary=summary.reset_index(); summary.to_csv(out/"repeated_outer_cascade_summary.csv",index=False)
    # Config selection frequency and per-stage AP summary.
    chosen_counts=stage_sel.groupby(["precision_floor","stage","config","feature_set"]).size().reset_index(name="selection_count")
    chosen_counts.to_csv(out/"selected_config_frequency.csv",index=False)
    ap_summary=apdf.groupby(["stage","config"]).oof_ap.agg(["mean","std","min","max"]).reset_index().sort_values(["stage","mean"],ascending=[True,False])
    ap_summary.to_csv(out/"inner_oof_ap_summary.csv",index=False)
    report=["# Repeated grouped benchmark: boosted models, oversampling and feature ablations","",
      "## Design","- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.","- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.","- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.","- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.","- Feature ablations: compact levels/recency/engagement; + slopes; + per-lesson features (trends and module families are tested separately). A remains demographics/signup-calendar only.", f"- Compact-only mode: {args.compact_only}.","- Families: logistic regression (unweighted and class-weighted), and GradientBoosting. Student-group oversampling is compared with no oversampling for logistic and GradientBoosting.",f"- The top {args.top_k} stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.",f"- Outer split count in this run: {len(args.outer_seeds)}. Repeated outer splits can overlap in students and are not independent external validation.","","## Repeated outer cascade summary",summary.to_markdown(index=False),"","## Selected configuration frequency",chosen_counts.to_markdown(index=False),"","## Inner OOF AP by stage/configuration (top rows)",ap_summary.groupby("stage",group_keys=False).head(12).to_markdown(index=False),"","## Important limitations","- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.","- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.","- B and C cascade scores use the maximum probability across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.","- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.","- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.","- This is a small controlled benchmark, not a broad hyperparameter search. Model configurations are intentionally conservative."]
    (out/"BENCHMARK_REPORT.md").write_text("\n".join(report)+"\n")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("Outputs:",out)

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    main()
