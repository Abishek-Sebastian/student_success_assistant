#!/usr/bin/env python3
"""Evaluate unlabeled explanations belonging to Task 1 flagged at-risk students.

Example:
  python scripts/task2/task2_run_at_risk.py \
    --prediction-file outputs/eventual_dropout_aligned/test_predictions_precision_50.csv \
    --limit 5

Only unlabeled explanations are selected so this demo does not reuse the held-out
human-labelled golden examples. If timestamps are absent, the script does not claim
that an explanation is the latest; it chooses one row per student deterministically.
"""
from pathlib import Path
import argparse, json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'task2'))
from src.evaluator import load_knowledge,evaluate_one


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prediction-file',default='outputs/eventual_dropout_aligned/test_predictions_precision_50.csv')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--output',default='task2/outputs/at_risk_examples.csv')
    a=p.parse_args()
    pred_path=ROOT/a.prediction_file
    if not pred_path.exists(): raise FileNotFoundError(f'Task 1 prediction file not found: {pred_path}')
    preds=pd.read_csv(pred_path)
    flag_col=next((c for c in ['cascade_flag','flag','prediction_at_primary_precision_floor'] if c in preds.columns),None)
    if flag_col is None: raise ValueError(f'No supported risk flag column in {preds.columns.tolist()}')
    flagged=preds[pd.to_numeric(preds[flag_col],errors='coerce').fillna(0).astype(int)==1] # type: ignore
    explanations=pd.read_csv(ROOT/'data/explanations.csv')
    # Restrict demo to unlabeled explanations to keep the locked golden set out of this workflow.
    explanations=explanations[explanations.label.isna()].copy()
    candidates=explanations[explanations.student_id.astype(str).isin(flagged.student_id.astype(str))]
    candidates=candidates.drop_duplicates('student_id',keep='first').head(a.limit)
    if candidates.empty:
        raise RuntimeError('No flagged students with unlabeled explanations were found. Check the prediction file and its risk flag.')
    knowledge=load_knowledge(); results=[]
    for _,row in candidates.iterrows():
        result=evaluate_one(row.to_dict(),knowledge)
        result['student_id']=row.student_id
        result['lesson_id']=row.lesson_id
        result['explanation_text']=row.explanation_text
        results.append(result)
        print(f"{row.student_id} / {row.explanation_id}: {result['verdict']} ({result['status']})")
    out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out,index=False)
    (out.with_suffix('.json')).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(f'Wrote {len(results)} examples to {out}')

if __name__=='__main__': main()
