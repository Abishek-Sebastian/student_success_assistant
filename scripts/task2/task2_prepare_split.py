#!/usr/bin/env python3
"""Prepare all 60 supplied labelled explanations as the Task 2 golden set.

No labelled row is designated as development data or used as episodic memory.
Inputs and labels are written separately so labels are only joined after inference.
"""
from pathlib import Path
import hashlib, json
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/explanations.csv'
OUT = ROOT / 'task2/data/evaluation'

def main():
    df = pd.read_csv(SOURCE)
    df['label'] = df['label'].fillna('').astype(str).str.strip().str.lower()
    labelled = df[df.label.ne('')].copy()
    allowed = {'understood', 'partial', 'wrong'}
    if len(labelled) != 60 or set(labelled.label) != allowed:
        raise SystemExit(f'Expected 60 labelled rows with labels {allowed}; got {len(labelled)} rows and {set(labelled.label)}')
    cols = ['explanation_id', 'student_id', 'lesson_id', 'explanation_text']
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = labelled[cols].sort_values('explanation_id')
    labels = labelled[['explanation_id', 'label']].sort_values('explanation_id')
    inputs.to_csv(OUT/'golden_inputs.csv', index=False)
    labels.to_csv(OUT/'golden_labels.csv', index=False)
    input_hash = hashlib.sha256((OUT/'golden_inputs.csv').read_bytes()).hexdigest()
    label_hash = hashlib.sha256((OUT/'golden_labels.csv').read_bytes()).hexdigest()
    manifest = {
      'split_version':'all-labelled-v1',
      'method':'all 60 provided human-labelled rows are the golden evaluation set; no development split',
      'total_labelled_rows':len(labelled), 'development_rows':0, 'golden_rows':len(labelled),
      'golden_labels_are_separate_file':True,
      'class_counts':labels.label.value_counts().sort_index().to_dict(),
      'golden_inputs_sha256':input_hash, 'golden_labels_sha256':label_hash,
      'memory_policy':'No row or label from this set may be used as episodic memory.',
      'evaluation_policy':'Repeated evaluation after inspecting results is iterative benchmark tracking, not an untouched test estimate.'
    }
    (ROOT/'task2/data/split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__': main()
