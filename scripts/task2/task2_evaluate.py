#!/usr/bin/env python3
"""Run Task 2 predictions, then score against the locked human-labelled golden set."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'task2'))
from src.evaluator import load_knowledge, evaluate_one, VALID_STATUSES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['golden','development'],default='development')
    ap.add_argument('--limit',type=int,default=None,help='Smoke-test only; do not use limited runs as final metrics')
    ap.add_argument('--output-dir',default=None)
    ap.add_argument('--confirm-final-evaluation',action='store_true',help='Required to run the locked golden evaluation')
    ap.add_argument('--overwrite-final',action='store_true',help='Allow replacing existing golden prediction artifacts; avoid unless explicitly necessary')
    args=ap.parse_args()
    if args.split == 'golden' and not args.confirm_final_evaluation:
        ap.error('Golden evaluation is locked. Add --confirm-final-evaluation only after freezing model, prompt, memory, and retry settings.')
    if args.split == 'golden' and args.limit is not None:
        ap.error('Do not use --limit for final golden evaluation.')
    if args.split=='golden':
        inputs=ROOT/'task2/data/evaluation/golden_inputs.csv'
        labels=ROOT/'task2/data/evaluation/golden_labels.csv'
    else:
        inputs=ROOT/'task2/data/development/development_cases.csv'
        labels=inputs
    manifest_path=ROOT/'task2/data/split_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if args.split == 'golden':
        actual_input_hash=hashlib.sha256(inputs.read_bytes()).hexdigest()
        if manifest.get('golden_inputs_sha256') and actual_input_hash != manifest['golden_inputs_sha256']:
            raise SystemExit('Golden inputs hash does not match split manifest; refusing final evaluation.')
    df=pd.read_csv(inputs).fillna('')
    # The development CSV contains labels for scoring, but never pass them to inference.
    df=df.drop(columns=['label'],errors='ignore')
    if args.limit: df=df.head(args.limit)
    outdir=Path(args.output_dir) if args.output_dir else ROOT/'task2/outputs'/args.split
    if args.split == 'golden' and (outdir/'predictions.csv').exists() and not args.overwrite_final:
        existing_predictions = outdir / "predictions.csv"
        raise SystemExit(f"Golden prediction file already exists at {existing_predictions}. Refusing to overwrite; use a fresh output directory or --overwrite-final deliberately.")
    knowledge=load_knowledge(); results=[]
    for i,row in df.iterrows():
        print(f"[{len(results)+1}/{len(df)}] {row['explanation_id']} ({row['lesson_id']})",flush=True)
        result=evaluate_one(row.to_dict(),knowledge)
        results.append(result)
        print(f"  verdict={result['verdict']} status={result['status']} latency={result.get('latency_seconds',0)}s",flush=True)
    outdir.mkdir(parents=True,exist_ok=True)
    pred=pd.DataFrame(results)
    pred.to_csv(outdir/'predictions.csv',index=False)
    pred.to_json(outdir/'predictions.jsonl',orient='records',lines=True,force_ascii=False)
    summary={'split':args.split,'n_predictions':len(pred),'valid_model_outputs':int(pred.status.isin(VALID_STATUSES).sum()),'failed_outputs':int((pred.status=='failed_after_retries').sum()),'empty_input_fallbacks':int((pred.status=='empty_input_fallback').sum()),'mean_latency_seconds':float(pred.latency_seconds.mean()) if len(pred) else None,'median_latency_seconds':float(pred.latency_seconds.median()) if len(pred) else None,'model':__import__('os').getenv('LLM_MODEL','unsloth/Qwen3.5-4B-Instruct'),'base_url':__import__('os').getenv('LLM_BASE_URL','http://localhost:8000/v1'),'limited_run':args.limit is not None}
    repairs = pred.get('feedback_field_repairs', pd.Series([[] for _ in range(len(pred))]))
    summary['predictions_with_repaired_feedback_fields'] = int(repairs.map(lambda x: bool(x) if isinstance(x, list) else bool(x) and str(x) not in ('[]', 'nan')).sum())
    summary['total_repaired_feedback_fields'] = int(repairs.map(lambda x: len(x) if isinstance(x, list) else len([v for v in str(x).strip('[]').split(',') if v.strip()]) if str(x).strip() not in ('', '[]', 'nan') else 0).sum())
    summary['total_model_attempts'] = int(pred.get('attempt_count', pd.Series([0] * len(pred))).fillna(0).sum())
    summary['retry_recoveries'] = int(pred.status.astype(str).str.startswith('recovered_after_retry').sum())
    summary['json_schema_failure_count'] = int((pred.status == 'failed_after_retries').sum())
    # Critical protocol: predictions are fully generated and written before the label file is read.
    if args.split == 'golden':
        actual_label_hash=hashlib.sha256(labels.read_bytes()).hexdigest()
        if manifest.get('golden_labels_sha256') and actual_label_hash != manifest['golden_labels_sha256']:
            raise SystemExit('Golden labels hash does not match split manifest; predictions saved but scoring refused.')
    gold=pd.read_csv(labels)
    scored=pred[['explanation_id','verdict','status']].merge(gold[['explanation_id','label']],on='explanation_id',how='inner',validate='one_to_one')
    if len(scored)!=len(pred): raise RuntimeError('Prediction/label ID mismatch; refusing to score')
    # Failed or empty inputs are unscorable, not automatically classified as "wrong".
    valid=scored[scored.status.isin(VALID_STATUSES) & scored.verdict.isin(['wrong','partial','understood'])]
    summary['valid_prediction_count']=int(len(valid))
    summary['scorable_coverage']=float(len(valid)/len(scored)) if len(scored) else None
    summary['valid_prediction_accuracy']=float(accuracy_score(valid.label,valid.verdict)) if len(valid) else None
    summary['classification_report_valid_predictions']=classification_report(valid.label,valid.verdict,labels=['wrong','partial','understood'],output_dict=True,zero_division=0) if len(valid) else None
    summary['confusion_matrix_labels_order']=['wrong','partial','understood']
    summary['confusion_matrix_valid_predictions']=confusion_matrix(valid.label,valid.verdict,labels=['wrong','partial','understood']).tolist()
    summary['per_class_report_text_valid_predictions']=classification_report(valid.label,valid.verdict,labels=['wrong','partial','understood'],zero_division=0) if len(valid) else 'No valid predictions.'
    summary['evaluation_note']='Accuracy is computed only on valid model predictions. Feedback-field fallbacks are explicitly counted and must be included when assessing feedback quality. Failed/empty cases are unscorable and reported through failure counts and scorable_coverage.'
    scored.to_csv(outdir/'scored_predictions.csv',index=False)
    (outdir/'metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if args.split=='golden':
        print('\nFINAL EVALUATION NOTE: This golden set is now considered consumed for this model/configuration. Do not tune on this result and re-report it as untouched. Use a new held-out set for any subsequent final claim.')
if __name__=='__main__': main()
