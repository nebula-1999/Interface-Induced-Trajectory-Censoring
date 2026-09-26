#!/usr/bin/env python3
"""Offline, answer-blind JSON recovery; not the deployed dedicated parser.

Usage: --gorilla PATH to the pinned checkout to create an anonymous evidence bundle.
Then --bundle PATH recomputes it using only the bundle and Python stdlib.
No generated program is executed. Official AST scoring checks answer correctness.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import types

PIN = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
ROOT = Path(__file__).resolve().parents[1]
MODEL = 'P1-Qwen2.5-Coder-7B-Hermes-FC'

def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def reject_constant(value):
    raise ValueError('Non-JSON numeric constant: '+value)

def recover(text):
    """Left-to-right maximal JSON values; retain every call, without gold filtering.

    Accept name/arguments or a function wrapper; string arguments must decode to
    an object. Never fix malformed JSON, names or values. Repeated calls remain.
    """
    decoder = json.JSONDecoder(parse_constant=reject_constant)
    calls = []
    i = 0
    while i < len(text):
        if text[i] not in '{[':
            i += 1
            continue
        try:
            value, length = decoder.raw_decode(text[i:])
        except ValueError:
            i += 1
            continue
        i += length
        for obj in value if isinstance(value, list) else [value]:
            if not isinstance(obj, dict):
                continue
            obj = obj.get('function', obj)
            if not isinstance(obj, dict):
                continue
            name, args = obj.get('name'), obj.get('arguments')
            if isinstance(args, str):
                try: args = json.loads(args,parse_constant=reject_constant)
                except ValueError: continue
            if isinstance(name, str) and isinstance(args, dict):
                calls.append({'name': name, 'arguments': args})
    return calls

def structural_schema(value, schema):
    """Types/required/properties/items/enum; format is an annotation, not checked.

    Unknown constraints abort instead of silently passing. Extra keys are rejected
    as an explicit tool-signature rule, even without additionalProperties:false.
    """
    allowed = {'type','properties','required','items','enum','description','format','default','optional'}
    assert not set(schema)-allowed, set(schema)-allowed
    typ = schema['type']
    checks = {'object': isinstance(value, dict), 'array': isinstance(value, list),
              'string': isinstance(value, str), 'boolean': type(value) is bool,
              'integer': type(value) is int, 'number': type(value) in (int,float)}
    assert typ in checks, typ
    if not checks[typ] or ('enum' in schema and value not in schema['enum']): return False
    if typ == 'object':
        props = schema.get('properties', {})
        return (set(schema.get('required', [])) <= set(value) <= set(props)
                and all(structural_schema(v, props[k]) for k,v in value.items()))
    if typ == 'array': return all(structural_schema(v,schema['items']) for v in value)
    return True

def load_scorer(bundle):
    # Avoid importing unrelated online inference SDKs. This sole registry property
    # matches p1/bfcl_registration.py; all scoring source files remain byte-identical.
    sys.path.insert(0, str(bundle/'vendor'))
    registry = types.ModuleType('bfcl_eval.constants.model_config')
    registry.MODEL_CONFIG_MAPPING = {MODEL: types.SimpleNamespace(underscore_to_dot=True)}
    sys.modules[registry.__name__] = registry
    checker = importlib.import_module('bfcl_eval.eval_checker.ast_eval.ast_checker')
    return checker.ast_checker, checker.Language.PYTHON

def build(gorilla, bundle):
    base = gorilla/'berkeley-function-call-leaderboard'
    bundle.mkdir(parents=True, exist_ok=True)
    sources = ['constants/enums.py','constants/type_mappings.py',
        'eval_checker/ast_eval/ast_checker.py',
        'eval_checker/ast_eval/type_convertor/java_type_converter.py',
        'eval_checker/ast_eval/type_convertor/js_type_converter.py']
    hashes = {}
    for name in sources:
        src = base/'bfcl_eval'/name
        dst = bundle/'vendor/bfcl_eval'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src,dst)
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copyfile(gorilla/'LICENSE',bundle/'vendor/LICENSE')
    data = {r['id']:r for r in read_rows(base/'bfcl_eval/data/BFCL_v4_simple_python.json')}
    gold = {r['id']:r for r in read_rows(base/'bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json')}
    proxy = read_rows(ROOT/'runs/p1_clean/logs/bfcl_hermes.jsonl')
    proxy = {r['case_id']:r for r in proxy if r['request_index']==0 and r['case_id'].startswith('simple_python_')}
    records = []
    for arm in ['hermes','repaired']:
        directory = ROOT/f'runs/p1_clean/bfcl_run/full_{arm}/result'
        results = read_rows(next(directory.rglob('BFCL_v4_simple_python_result.json')))
        for r in sorted(results,key=lambda r:r['id']):
            inp = r['inference_log'][0]['content']
            cid = r['id']
            rec = {'arm':arm,'case_id':cid,'request':inp,'result':r['result'],
                   'function':data[cid]['function'],'ground_truth':gold[cid]['ground_truth']}
            if arm=='hermes':
                p=proxy[cid]
                assert p['content']==r['result'] and not p['parsed_tool_calls']
                rec.update(raw_output=p['content'],raw_sha256=hashlib.sha256(p['content'].encode()).hexdigest(),
                           server_parsed=p['parsed_tool_calls'],status=p['status'])
            records.append(rec)
    (bundle/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (bundle/'provenance.json').write_text(json.dumps({'bfcl_commit':PIN,'scorer_sha256':hashes,
        'source_run':'p1_clean','registry_underscore_to_dot':True,
        'scope':'100 simple_python cases; no multi-turn counterfactual scoring',
        'request_boundary':'BFCL inference_input log; not a full HTTP request capture',
        'schema_boundary':'structural signature checks; format annotations not validated'},indent=2)+'\n')

def evaluate(bundle):
    provenance=json.loads((bundle/'provenance.json').read_text())
    for name, expected in provenance['scorer_sha256'].items():
        assert hashlib.sha256((bundle/'vendor/bfcl_eval'/name).read_bytes()).hexdigest()==expected
    scorer, language=load_scorer(bundle)
    details=[]
    for r in read_rows(bundle/'records.jsonl'):
        raw=r.get('raw_output')
        if raw is not None:
            assert hashlib.sha256(raw.encode()).hexdigest()==r['raw_sha256']
            calls=recover(raw)
        else:
            calls=([{'name':k,'arguments':json.loads(v)} for c in r['result'] for k,v in c.items()]
                   if isinstance(r['result'],list) else [])
        offered={f['function']['name']:f['function']['parameters'] for f in r['request']['tools']}
        names_ok=bool(calls) and all(c['name'] in offered for c in calls)
        schema_ok=names_ok and all(structural_schema(c['arguments'],offered[c['name']]) for c in calls)
        decoded=[{c['name']:c['arguments']} for c in calls]
        verdict=scorer(r['function'],decoded,r['ground_truth'],language,'simple_python',MODEL)
        details.append(dict(arm=r['arm'],case_id=r['case_id'],calls=calls,
            recovered=bool(calls),allowed_names=names_ok,structural_schema=schema_ok,
            official_ast_valid=verdict['valid'],official_verdict=verdict,
            has_dedicated_envelope=bool(raw and '<tools>' in raw)))
    summary={}
    for arm in ['hermes','repaired']:
        rows=[r for r in details if r['arm']==arm]
        summary[arm]={'n':len(rows),**{k:sum(r[k] for r in rows) for k in
            ['recovered','allowed_names','structural_schema','official_ast_valid','has_dedicated_envelope']}}
    assert summary['repaired']['official_ast_valid']==96, 'Official scorer positive control changed'
    assert summary['hermes']['n']==100
    (bundle/'replay_details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    (bundle/'replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gorilla',type=Path)
    parser.add_argument('--bundle',type=Path,default=ROOT/'submission_packages/mechanism_evidence/bfcl')
    args=parser.parse_args()
    if args.gorilla: build(args.gorilla,args.bundle)
    evaluate(args.bundle)
