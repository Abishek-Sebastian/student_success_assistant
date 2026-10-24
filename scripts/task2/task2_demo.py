#!/usr/bin/env python3
"""Run a single explanation through the Task 2 evaluator."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'task2'))
from src.evaluator import load_knowledge,evaluate_one

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lesson-id',required=True,choices=['L1','L2','L3'])
    p.add_argument('--explanation',required=True)
    p.add_argument('--explanation-id',default='DEMO_001')
    a=p.parse_args()
    print(json.dumps(evaluate_one({'lesson_id':a.lesson_id,'explanation_text':a.explanation,'explanation_id':a.explanation_id},load_knowledge()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
