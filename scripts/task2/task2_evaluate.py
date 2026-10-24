#!/usr/bin/env python3
"""Run Task 2 predictions, then score against the locked human-labelled golden set."""
from pathlib import Path
import argparse, json, sys, time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'task2'))
from src.evaluator import load_knowledge, evaluate_one


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['golden','development'],default='golden')
    ap.add_argument('--limit',type=int,default=None,help='Smoke-test only; do not use limited runs as final metrics')
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args()
    if args.split=='golden':
        inputs=ROOT/'task2/data/evaluation/golden_inputs.csv'
        labels=ROOT/'task2/data/evaluation/golden_labels.csv'
    else:
        inputs=ROOT/'task2/data/development/development_cases.csv'
        labels=inputs
    df=pd.read_csv(inputs).fillna('')
    if args.limit: df=df.head(args.limit)
    knowledge=load_knowledge(); results=[]
    for i,row in df.iterrows():
        print(f"[{len(results)+1}/{len(df)}] {row['explanation_id']} ({row['lesson_id']})",flush=True)
        result=evaluate_one(row.to_dict(),knowledge)
        results.append(result)
        print(f"  verdict={result['verdict']} status={result['status']} latency={result.get('latency_seconds',0)}s",flush=True)
    outdir=Path(args.output_dir) if args.output_dir else ROOT/'task2/outputs'/args.split
    outdir.mkdir(parents=True,exist_ok=True)
    pred=pd.DataFrame(results)
    pred.to_csv(outdir/'predictions.csv',index=False)
    pred.to_json(outdir/'predictions.jsonl',orient='records',lines=True,force_ascii=False)
    summary={'split':args.split,'n_predictions':len(pred),'valid_model_outputs':int(pred.status.isin(['ok','recovered_after_retry']).sum()),'failed_outputs':int((pred.status=='failed_after_retries').sum()),'empty_input_fallbacks':int((pred.status=='empty_input_fallback').sum()),'predictions_with_repaired_feedback_fields':int(pred.get('feedback_field_repairs',pd.Series([[]]*len(pred))).map(lambda x: bool(x) if isinstance(x,list) else bool(x) and x not in ('[]','')).sum()),'total_repaired_feedback_fields':int(pred.get('feedback_field_repairs',pd.Series([[]]*len(pred))).map(lambda x: len(x) if isinstance(x,list) else len(json.loads(x)) if isinstance(x,str) and x.startswith('[') else 0).sum()),'mean_latency_seconds':float(pred.latency_seconds.mean()) if len(pred) else None,'median_latency_seconds':float(pred.latency_seconds.median()) if len(pred) else None,'model':__import__('os').getenv('LLM_MODEL','unsloth/Qwen3.5-4B-Instruct'),'base_url':__import__('os').getenv('LLM_BASE_URL','http://localhost:8000/v1'),'limited_run':args.limit is not None}
    # Labels are joined only after all predictions have been generated and written.
    gold=pd.read_csv(labels)
    scored=pred[['explanation_id','verdict','status']].merge(gold[['explanation_id','label']],on='explanation_id',how='inner',validate='one_to_one')
    if len(scored)!=len(pred): raise RuntimeError('Prediction/label ID mismatch; refusing to score')
    # Failed or empty inputs are unscorable, not automatically classified as "wrong".
    valid=scored[scored.status.isin(['ok','recovered_after_retry']) & scored.verdict.isin(['wrong','partial','understood'])]
    summary['valid_prediction_count']=int(len(valid))
    summary['scorable_coverage']=float(len(valid)/len(scored)) if len(scored) else None
    summary['valid_prediction_accuracy']=float(accuracy_score(valid.label,valid.verdict)) if len(valid) else None
    summary['classification_report_valid_predictions']=classification_report(valid.label,valid.verdict,labels=['wrong','partial','understood'],output_dict=True,zero_division=0) if len(valid) else None
    summary['confusion_matrix_labels_order']=['wrong','partial','understood']
    summary['confusion_matrix_valid_predictions']=confusion_matrix(valid.label,valid.verdict,labels=['wrong','partial','understood']).tolist()
    summary['per_class_report_text_valid_predictions']=classification_report(valid.label,valid.verdict,labels=['wrong','partial','understood'],zero_division=0) if len(valid) else 'No valid predictions.'
    summary['evaluation_note']='Accuracy is computed only on valid model predictions. Failed/empty cases are unscorable and reported through failure counts and scorable_coverage.'
    scored.to_csv(outdir/'scored_predictions.csv',index=False)
    (outdir/'metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.split=='golden' and args.limit is None:
        print('\nFINAL EVALUATION NOTE: Do not tune on this result and re-report it as an untouched final score. Freeze the run artifacts and use a new held-out set for any new final evaluation.')
if __name__=='__main__': main()
