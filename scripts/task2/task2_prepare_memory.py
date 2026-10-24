#!/usr/bin/env python3
"""Create an empty-by-default episodic memory, never from the supplied labelled set.

Only independently human-verified examples from outside the 60-row golden set may be
added manually in a future version. This script deliberately does not import dataset rows.
"""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = ROOT/'task2/memory'
OUT = MEMORY_DIR/'approved_cases.jsonl'
MANIFEST = MEMORY_DIR/'case_memory_manifest.json'

def main():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT.exists(): OUT.write_text('', encoding='utf-8')
    records=[]
    for line in OUT.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        record=json.loads(line)
        if record.get('source_dataset') == 'provided_60_labelled_rows' or record.get('split') == 'golden':
            raise SystemExit('Refusing memory containing a row from the 60-row golden dataset')
        if record.get('approval_status') != 'approved' or record.get('label_source') not in {'manual_review', 'larger_llm_then_human_verified'}:
            raise SystemExit('Every memory case must be explicitly approved and independently verified')
        records.append(record)
    manifest={
      'memory_version':2, 'source':'independently verified cases only; never task2/data/explanations.csv',
      'case_count':len(records), 'label_sources_allowed':['manual_review','larger_llm_then_human_verified'],
      'golden_data_used':False, 'provided_labelled_rows_allowed':False,
      'model_generated_cases_allowed_without_human_verification':False,
      'retention_policy':'No automatic retention. Each case must be verified and manually approved outside the golden dataset.',
      'memory_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
