"""Evaluate provenance detection with HUMAN-labeled assertions across papers.

Gold JSONL rows: {"paper_id":"...","evidence_id":"P0001:s1",
                  "claim_origin":"current_study|cited_study|undetermined|..."}
Prediction JSONL rows: same paper_id,evidence_id and claim_origin; optionally
one row per assertion_id when clauses are aligned and reviewed.

This script never generates gold labels. Use it on independently reviewed papers
for precision, recall, confusion matrices and coverage, not calibration guesses.
"""
from collections import Counter
from pathlib import Path
import argparse,json

def read_jsonl(path):
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        if not all(k in row for k in ('paper_id','evidence_id','claim_origin')):
            raise ValueError(f'{path}:{i}: missing paper_id/evidence_id/claim_origin')
        rows.append(row)
    return rows

def evaluate(gold,pred):
    def key(r):return (r['paper_id'],r['evidence_id'],r.get('assertion_id',''))
    a={key(r):r['claim_origin'] for r in gold}
    b={key(r):r['claim_origin'] for r in pred}
    keys=sorted(set(a)&set(b)); labels=sorted(set(a.values())|set(b.values()))
    confusion=Counter((a[k],b[k]) for k in keys)
    report=[]
    for label in labels:
        tp=confusion[label,label]
        fp=sum(n for (g,p),n in confusion.items() if p==label and g!=label)
        fn=sum(n for (g,p),n in confusion.items() if g==label and p!=label)
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        report.append({'label':label,'support':sum(a[k]==label for k in keys),'precision':precision,'recall':recall})
    return {'gold_records':len(a),'predictions':len(b),'aligned_records':len(keys),
            'coverage':len(keys)/len(a) if a else 0,'by_origin':report,
            'confusion':[{'gold':g,'predicted':p,'count':n} for (g,p),n in sorted(confusion.items())],
            'macro_precision':sum(x['precision'] for x in report if x['precision'] is not None)/max(1,sum(x['precision'] is not None for x in report)),
            'macro_recall':sum(x['recall'] for x in report if x['recall'] is not None)/max(1,sum(x['recall'] is not None for x in report))}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold',required=True);ap.add_argument('--predicted',required=True)
    ap.add_argument('--out')
    args=ap.parse_args()
    result=evaluate(read_jsonl(args.gold),read_jsonl(args.predicted))
    output=json.dumps(result,indent=2)
    if args.out:Path(args.out).write_text(output,encoding='utf-8')
    print(output)
