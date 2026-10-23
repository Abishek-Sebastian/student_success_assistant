#!/usr/bin/env python3
"""Train stage models on a shared eventual-dropout label and evaluate the cascade.

All features are restricted to the observed prefix. Student IDs are grouped in
CV and held out as a unit. B/C include final observed prefixes; those snapshots
are labelled with the student's eventual outcome rather than censored/negative.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ["age_group", "country", "device", "signup_month", "signup_weekday"]
ATTEMPT = ["score_valid_count", "score_missing_or_invalid_fraction", "score_mean", "score_last_valid", "score_last3_mean", "time_valid_count", "time_missing_fraction", "time_mean", "time_last_valid", "time_last3_mean", "time_total_valid_sum", "mean_gap_hours", "last_gap_hours", "history_span_hours"]
MODELS = ["dummy_prior", "logistic_unweighted", "logistic_balanced", "extra_trees_balanced"]

def feature_frame(df, stage):
    x = df.drop(columns=["student_id", "target", "_row_id"], errors="ignore").copy()
    if "signup_date" in x:
        d = pd.to_datetime(x.pop("signup_date"), errors="coerce")
        x["signup_month"] = d.dt.month.astype(float)
        x["signup_weekday"] = d.dt.dayofweek.astype(float)
    cols = [c for c in (CATS if stage == "A" else CATS + ATTEMPT) if c in x.columns]
    x = x[cols]
    cats = [c for c in CATS if c in x.columns]
    nums = [c for c in x.columns if c not in cats]
    return x, cats, nums

def pipeline(cats, nums, model):
    transformers=[]
    if nums:
        steps=[("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if model in ("logistic_unweighted", "logistic_balanced"):
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), nums))
    if cats:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats))
    prep=ColumnTransformer(transformers, remainder="drop")
    if model == "dummy_prior": est=DummyClassifier(strategy="prior")
    elif model == "logistic_unweighted": est=LogisticRegression(max_iter=3000, random_state=42)
    elif model == "logistic_balanced": est=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif model == "extra_trees_balanced": est=ExtraTreesClassifier(n_estimators=40, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=1)
    else: raise ValueError(model)
    return Pipeline([("prep", prep), ("model", est)])

def fit_pipe(pipe, X, y, student_ids):
    # Each student's total contribution is 1, avoiding over-weighting students
    # simply because they have more snapshots. Class weights remain model-specific.
    counts=pd.Series(student_ids).value_counts()
    weights=np.array([1.0/counts[s] for s in student_ids], dtype=float)
    pipe.fit(X, y, model__sample_weight=weights)
    return pipe

def choose_student_split(meta, test_size, seed):
    ids=meta.student_id.astype(str).to_numpy(); y=meta.eventual_dropout.to_numpy()
    sp=StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr,te=next(sp.split(ids.reshape(-1,1),y))
    return set(ids[tr]),set(ids[te])

def make_stage_data(input_dir, audit):
    # A: one row for every student, including those with no attempts.
    a=pd.read_csv(input_dir/"A_model_table.csv")
    a=a.drop(columns=["target"], errors="ignore").merge(audit[["student_id","eventual_dropout"]], on="student_id", how="left", validate="one_to_one").rename(columns={"eventual_dropout":"target"})
    a["_row_id"]=np.arange(len(a))
    # B: prefixes 1 and 2, including final/censored prefixes. Re-label all with D.
    b_parts=[pd.read_csv(input_dir/"B_rolling_early.csv"), pd.read_csv(input_dir/"B_rolling_early_censored_rows.csv")]
    b=pd.concat(b_parts, ignore_index=True).drop_duplicates(["student_id","snapshot_index"])
    b=b.drop(columns=["target","target_observed","target_reason_audit","eventual_attempt_count_audit","prefix_length_audit","next_attempt_available_audit"], errors="ignore")
    b=b.merge(audit[["student_id","eventual_dropout"]], on="student_id", how="left", validate="many_to_one").rename(columns={"eventual_dropout":"target"})
    b["_row_id"]=np.arange(len(b))
    # C: every prefix k>=3, including final observed prefixes.
    c_parts=[pd.read_csv(input_dir/"C2_rolling.csv"), pd.read_csv(input_dir/"C2_censored_rows.csv")]
    c=pd.concat(c_parts, ignore_index=True).drop_duplicates(["student_id","snapshot_index"])
    c=c.drop(columns=["target","target_observed","target_reason_audit","eventual_attempt_count_audit","prefix_length_audit","next_attempt_available_audit"], errors="ignore")
    c=c.merge(audit[["student_id","eventual_dropout"]], on="student_id", how="left", validate="many_to_one").rename(columns={"eventual_dropout":"target"})
    c["_row_id"]=np.arange(len(c))
    for name,df in [("A",a),("B",b),("C",c)]:
        if df.target.isna().any(): raise ValueError(f"Unmatched outcome for {name}")
        df.target=df.target.astype(int)
    return {"A":a,"B":b,"C":c}

def max_student_scores(df, scores, all_ids):
    tmp=pd.DataFrame({"student_id":df.student_id.astype(str).to_numpy(),"score":scores})
    return tmp.groupby("student_id").score.max().reindex(all_ids).to_numpy(dtype=float)

def threshold_candidates(scores, n=9):
    x=np.asarray(scores,float); x=x[np.isfinite(x)]
    if not len(x): return [float("inf")]
    return sorted(set(np.quantile(x,np.linspace(0,1,n)).tolist()+[float(np.nextafter(x.max(),np.inf))]))

def policy_metrics(y, flags):
    y=np.asarray(y,int); flags=np.asarray(flags,bool); tn,fp,fn,tp=confusion_matrix(y,flags.astype(int),labels=[0,1]).ravel()
    return {"precision":float(precision_score(y,flags,zero_division=0)),"recall":float(recall_score(y,flags,zero_division=0)),"f1":float(f1_score(y,flags,zero_division=0)),"flagged":int(flags.sum()),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"n":int(len(y)),"positives":int(y.sum()),"prevalence":float(y.mean())}

def bootstrap_ci(y, flags, n_boot, seed):
    rng=np.random.default_rng(seed); y=np.asarray(y,int); flags=np.asarray(flags,bool); vals=[]
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    # Stratified student bootstrap keeps both classes represented in each replicate.
    for _ in range(n_boot):
        ix=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        yy=y[ix]; ff=flags[ix]
        tp=int(np.sum(ff & (yy==1))); nflag=int(ff.sum()); npos=int(np.sum(yy==1))
        vals.append([tp/nflag if nflag else 0.0, tp/npos if npos else 0.0])
    arr=np.asarray(vals)
    if not flags.any():
        p_low=p_high=float("nan")
    else:
        p_low=float(np.quantile(arr[:,0],.025)); p_high=float(np.quantile(arr[:,0],.975))
    return {"precision_ci_low":p_low,"precision_ci_high":p_high,"recall_ci_low":float(np.quantile(arr[:,1],.025)),"recall_ci_high":float(np.quantile(arr[:,1],.975))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="data/cascade_snapshots")
    ap.add_argument("--output-dir",default="outputs/eventual_dropout_aligned")
    ap.add_argument("--test-size",type=float,default=.20)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--precision-floors",type=float,nargs="+",default=[.30,.40,.50,.60])
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args(); root=Path.cwd(); inp=Path(args.input_dir); out=Path(args.output_dir)
    if not inp.is_absolute(): inp=root/inp
    if not out.is_absolute(): out=root/out
    out.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(inp/"snapshot_audit.csv")[["student_id","dropout_binary_audit_only"]].drop_duplicates("student_id")
    audit=audit.rename(columns={"dropout_binary_audit_only":"eventual_dropout"}); audit.student_id=audit.student_id.astype(str)
    meta=audit.copy(); meta.eventual_dropout=meta.eventual_dropout.astype(int)
    train_ids,test_ids=choose_student_split(meta,args.test_size,args.seed)
    if train_ids & test_ids: raise AssertionError("student split leakage")
    train_meta=meta[meta.student_id.isin(train_ids)].copy(); test_meta=meta[meta.student_id.isin(test_ids)].copy()
    train_ids_sorted=train_meta.student_id.astype(str).tolist(); test_ids_sorted=test_meta.student_id.astype(str).tolist()
    y_dev=train_meta.eventual_dropout.to_numpy(); y_test=test_meta.eventual_dropout.to_numpy()
    stages=make_stage_data(inp,audit)
    # Shared student folds over the development cohort.
    nfold=min(args.folds,int(y_dev.sum()),int((y_dev==0).sum()))
    cv=StratifiedGroupKFold(n_splits=nfold,shuffle=True,random_state=args.seed)
    fold_splits=list(cv.split(train_meta.student_id.to_numpy().reshape(-1,1),y_dev,groups=train_meta.student_id.to_numpy()))
    oof_scores={s:{m:np.full(len(train_meta),np.nan) for m in MODELS} for s in stages}
    stage_feature_info={}
    for stage,df in stages.items():
        print(f"Stage {stage}: {len(df)} snapshots", flush=True)
        X,cats,nums=feature_frame(df,stage); stage_feature_info[stage]=(X,cats,nums)
        row_sid=df.student_id.astype(str).to_numpy(); row_y=df.target.to_numpy()
        for model in MODELS:
            print(f"  OOF {stage} / {model}", flush=True)
            # Predict every validation student's available snapshots; aggregate max later.
            snapshot_oof=np.full(len(df),np.nan)
            for fold,(tr_i,va_i) in enumerate(fold_splits,1):
                fold_tr_ids=set(train_meta.student_id.iloc[tr_i].astype(str)); fold_va_ids=set(train_meta.student_id.iloc[va_i].astype(str))
                tr_mask=np.isin(row_sid,list(fold_tr_ids)); va_mask=np.isin(row_sid,list(fold_va_ids))
                if not va_mask.any(): continue
                fitted=fit_pipe(clone(pipeline(cats,nums,model)),X.loc[tr_mask],row_y[tr_mask],row_sid[tr_mask])
                snapshot_oof[va_mask]=fitted.predict_proba(X.loc[va_mask])[:,1]
            student_scores=max_student_scores(df,snapshot_oof,train_ids_sorted)
            oof_scores[stage][model]=student_scores
    # Pick stage models using OOF AP on students with stage checkpoints; selection is development-only.
    selection=[]; chosen={}; selected_oof={}
    for stage in stages:
        best=None
        for model in MODELS:
            scores=oof_scores[stage][model]; ok=np.isfinite(scores)
            apval=float(average_precision_score(y_dev[ok],scores[ok])) if len(np.unique(y_dev[ok]))==2 else np.nan
            selection.append({"stage":stage,"model":model,"development_oof_eventual_dropout_ap":apval,"students_scored":int(ok.sum()),"positives_scored":int(y_dev[ok].sum()),"positive_rate_scored":float(y_dev[ok].mean()) if ok.any() else np.nan})
            if np.isfinite(apval) and (best is None or apval>best[0]): best=(apval,model)
        chosen[stage]=best[1]; selected_oof[stage]=oof_scores[stage][best[1]]
    # Fit chosen models on development students, score held-out students at every available checkpoint.
    test_scores={}; pred_rows=[]
    for stage,df in stages.items():
        X,cats,nums=stage_feature_info[stage]; row_sid=df.student_id.astype(str).to_numpy()
        tr=np.isin(row_sid,list(train_ids)); te=np.isin(row_sid,list(test_ids))
        model=chosen[stage]; fitted=fit_pipe(clone(pipeline(cats,nums,model)),X.loc[tr],df.loc[tr,"target"].to_numpy(),row_sid[tr])
        raw=np.full(len(df),np.nan); raw[te]=fitted.predict_proba(X.loc[te])[:,1]
        test_scores[stage]=max_student_scores(df.loc[te].reset_index(drop=True),raw[te],test_ids_sorted)
    # Tune a joint OR cascade policy on development OOF student scores, maximize recall under precision floor.
    candidates={s:threshold_candidates(selected_oof[s]) for s in stages}
    print("Threshold candidate counts:", {k:len(v) for k,v in candidates.items()}, flush=True)
    # Precompute each stage's alert vector per threshold; the grid search then
    # combines boolean vectors rather than rebuilding score comparisons each time.
    flag_grid={s:[np.isfinite(selected_oof[s]) & (selected_oof[s]>=t) for t in candidates[s]] for s in stages}
    policy_rows=[]; pred_exports={}
    for floor in args.precision_floors:
        print(f"Threshold search precision floor {floor}", flush=True)
        best=None
        for ia,ta in enumerate(candidates["A"]):
            fa=flag_grid["A"][ia]
            for ib,tb in enumerate(candidates["B"]):
                fab=fa | flag_grid["B"][ib]
                for ic,tc in enumerate(candidates["C"]):
                    flags=fab | flag_grid["C"][ic]
                    n_flagged=int(flags.sum())
                    if n_flagged==0: continue
                    tp=int(np.sum(flags & (y_dev==1)))
                    precision=tp/n_flagged
                    if precision+1e-12<floor: continue
                    recall=tp/max(1,int(np.sum(y_dev==1)))
                    key=(recall,precision,-n_flagged)
                    if best is None or key>best[0]:
                        best=(key,(ta,tb,tc),{"precision":precision,"recall":recall,"flagged":n_flagged,"tp":tp})
        feasible=best is not None
        if feasible: thresholds=dict(zip(["A","B","C"],best[1])); devmet=best[2]
        else: thresholds={s:float("inf") for s in stages}; devmet=policy_metrics(y_dev,np.zeros(len(y_dev),bool))
        test_flags=np.zeros(len(test_meta),dtype=bool)
        for s in stages: test_flags |= np.isfinite(test_scores[s]) & (test_scores[s]>=thresholds[s])
        tm=policy_metrics(y_test,test_flags); ci=bootstrap_ci(y_test,test_flags,args.bootstrap,args.seed+int(floor*1000))
        policy_row={"precision_floor":floor,"development_feasible":feasible,"selected_model_A":chosen["A"],"selected_model_B":chosen["B"],"selected_model_C":chosen["C"],"threshold_A":thresholds["A"],"threshold_B":thresholds["B"],"threshold_C":thresholds["C"],**{f"development_{k}":v for k,v in devmet.items()},**{f"test_{k}":v for k,v in tm.items()},"test_precision_defined":bool(tm["flagged"]>0),**ci}
        if tm["flagged"]==0: policy_row["test_precision"]=float("nan")
        policy_rows.append(policy_row)
        exp=test_meta.copy()
        for s in stages:
            exp[f"stage_{s}_max_score"]=test_scores[s]
            exp[f"flag_{s}"]=np.isfinite(test_scores[s]) & (test_scores[s]>=thresholds[s])
        exp["cascade_flag"]=test_flags; exp["precision_floor"]=floor
        exp.to_csv(out/f"test_predictions_precision_{int(floor*100)}.csv",index=False)
        pred_exports[floor]=exp
    # Stage-level AP and availability coverage.
    pd.DataFrame(selection).to_csv(out/"stage_model_selection.csv",index=False)
    pd.DataFrame(policy_rows).to_csv(out/"end_to_end_precision_recall.csv",index=False)
    split_json={"seed":args.seed,"test_size":args.test_size,"train_student_ids":sorted(train_ids),"test_student_ids":sorted(test_ids)}
    (out/"split_ids.json").write_text(json.dumps(split_json,indent=2))
    ds=[]
    for s,df in stages.items():
        ds.append({"stage":s,"rows":len(df),"students":df.student_id.nunique(),"positive_snapshots":int(df.target.sum()),"positive_rate":float(df.target.mean()),"students_with_checkpoint":int(df.student_id.nunique()),"features_before_encoding":int(stage_feature_info[s][0].shape[1])})
    pd.DataFrame(ds).to_csv(out/"aligned_dataset_summary.csv",index=False)
    # Export development OOF student-level scores for reproducibility and future policy analyses.
    devexp=train_meta.copy()
    for s in stages: devexp[f"stage_{s}_oof_score"]=selected_oof[s]
    devexp.to_csv(out/"development_oof_student_scores.csv",index=False)
    # Report.
    p=pd.DataFrame(policy_rows); sel=pd.DataFrame(selection)
    lines=["# Eventual-dropout-aligned cascade experiment","",
      "## Objective","Train each stage to predict the same student-level eventual-dropout label, then choose stage thresholds jointly to maximize recall subject to a minimum precision constraint.","",
      "## Label and feature policy","- A: one row for every student, with demographics/signup-calendar features only and the eventual dropout label.","- B: all observed prefixes k=1 and k=2, including final observed prefixes; the target is eventual dropout, not next-attempt termination.","- C: all observed prefixes k>=3, including final observed prefixes; the target is eventual dropout.","- No attempt count, future prefix length, next-attempt availability, target-reason, or audit-only field is used as a predictor.","- Final observed prefixes are included because the historical eventual outcome is known for training; this does not mean the model knows future activity at inference time.","- Repeated snapshots are grouped by student for CV and holdout. Training weights make each student's snapshots sum to one per stage.","- B/C student scores for end-to-end evaluation are the maximum risk score over available checkpoints, matching the policy 'flag if any checkpoint crosses its threshold'.","",
      "## Dataset sizes",pd.DataFrame(ds).to_markdown(index=False),"",
      "## Stage model selection (development OOF AP)",sel.sort_values(["stage","development_oof_eventual_dropout_ap"],ascending=[True,False]).to_markdown(index=False),"",
      "## End-to-end precision-constrained recall",p.to_markdown(index=False),"",
      "## Interpretation and limitations","- Thresholds and model choices use development out-of-fold predictions only. The test set is student-level held out and is used once for evaluation.","- Bootstrap intervals resample held-out students stratified by outcome; they quantify test-cohort uncertainty, not all sources of model-selection uncertainty.","- With 500 students and few positive outcomes, results are highly split-sensitive. A precision floor met on development can fail on test.","- The supplied dropout label has no event timestamp. This experiment predicts eventual labelled dropout from observed prefixes, but cannot establish whether alerts precede a true dropout date or whether intervention changes outcomes.","- This is an OR-style alert policy over all observed checkpoints, not a claim that the deployment routing engine has been implemented or validated.","- A student with zero attempts has only the A score; B/C scores are unavailable."]
    (out/"EVENTUAL_DROPOUT_ALIGNED_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Selected models:",chosen)
    print(p[["precision_floor","development_precision","development_recall","test_precision","test_recall","test_flagged","test_tp","test_fp","test_fn","precision_ci_low","precision_ci_high","recall_ci_low","recall_ci_high"]].to_string(index=False))
    print("Outputs:",out)

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=UserWarning)
    main()
