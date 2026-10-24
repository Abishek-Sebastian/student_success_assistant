from pathlib import Path
import pandas as pd, re, unicodedata, json, itertools
root=Path(__file__).resolve().parents[2]
df=pd.read_csv(root/'data/explanations.csv')
lab=df[df['label'].notna()].copy()
lab['label']=lab['label'].str.strip().str.lower()
allowed={'understood','partial','wrong'}
assert set(lab.label)==allowed and len(lab)==60
# Normalize exact duplicate explanations so copies cannot cross development/evaluation.
def norm(s): return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s)).lower()).strip()
lab['_group']=lab.explanation_text.map(norm)
# Deterministically select whole text groups totaling 6 examples per class for development.
# Exhaustive combinations are cheap here (<=19 groups per class); choose the combination
# whose sorted group IDs are earliest after deterministic hash ordering.
import hashlib
dev_ids=[]
for label in sorted(allowed):
    groups=[]
    for text, g in lab[lab.label==label].groupby('_group', sort=False):
        gid=hashlib.sha256((label+'|'+text).encode()).hexdigest()
        groups.append((gid, text, list(g.explanation_id)))
    groups.sort(key=lambda z:z[0])
    found=None
    for r in range(1,len(groups)+1):
        for combo in itertools.combinations(groups,r):
            if sum(len(c[2]) for c in combo)==6:
                found=combo; break
        if found: break
    if found is None: raise RuntimeError(f'Cannot form 6-example dev split for {label}')
    dev_ids.extend(eid for c in found for eid in c[2])

dev=lab[lab.explanation_id.isin(dev_ids)].drop(columns=['_group'])
gold=lab[~lab.explanation_id.isin(dev_ids)].drop(columns=['_group'])
assert len(dev)==18 and len(gold)==42
assert (dev.label.value_counts().to_dict()=={'understood':6,'partial':6,'wrong':6})
assert set(dev.explanation_text.map(norm)).isdisjoint(set(gold.explanation_text.map(norm)))
# Keep development labels separate from evaluation inputs/labels.
cols=['explanation_id','student_id','lesson_id','explanation_text']
dev[cols+['label']].sort_values('explanation_id').to_csv(root/'task2/data/development/development_cases.csv',index=False)
gold[cols].sort_values('explanation_id').to_csv(root/'task2/data/evaluation/golden_inputs.csv',index=False)
gold[['explanation_id','label']].sort_values('explanation_id').to_csv(root/'task2/data/evaluation/golden_labels.csv',index=False)
# Metadata contains split sizes only, no gold labels.
import hashlib as _hashlib
input_bytes=(root/'task2/data/evaluation/golden_inputs.csv').read_bytes()
label_bytes=(root/'task2/data/evaluation/golden_labels.csv').read_bytes()
meta={'split_version':'v1','method':'deterministic hash-ordered exact-text-group split, class-stratified development subset','total_labelled_rows':60,'development_rows':len(dev),'golden_rows':len(gold),'development_per_class':dev.label.value_counts().sort_index().to_dict(),'golden_labels_are_separate_file':True,'exact_normalized_text_overlap':0,'golden_inputs_sha256':_hashlib.sha256(input_bytes).hexdigest(),'golden_labels_sha256':_hashlib.sha256(label_bytes).hexdigest(),'warning':'Do not inspect golden_labels.csv during prompt/model/retrieval development. Evaluation script joins it only after predictions are generated.'}
(root/'task2/data/split_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
