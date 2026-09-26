#!/usr/bin/env python3
"""Build a local anonymous submission package from an explicit allowlist.

No full rollout archive, checkpoints, git metadata, launch logs or public author
source is copied. Explicit raw-content subsets complement the statistical reduction.
"""
import json
from pathlib import Path
import re
import zipfile
from llama_paired_audit import FILES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'submission_packages'


def main():
    OUT.mkdir(exist_ok=True)
    files = {}
    def add(name, source):
        files[name] = (ROOT/source).read_bytes()
    for name in ['main.tex','main.pdf','main.bbl','abstract_openreview.txt']:
        add('paper/iclr2027/'+name,'paper/iclr2027/'+name)
    for dirname, glob in [('paper/iclr2027/sections','*.tex'),('paper/iclr2027/vendor','*'),
                          ('paper/sections','*.tex')]:
        for p in sorted((ROOT/dirname).glob(glob)):
            if p.is_file(): add(str(p.relative_to(ROOT)),p.relative_to(ROOT))
    for p in sorted((ROOT/'paper').glob('*.png')): add(str(p.relative_to(ROOT)),p.relative_to(ROOT))
    add('paper/refs.bib','paper/refs.bib')
    for name in ['submission_statistics.py','submission_statistics.json','llama_paired_audit.py',
                 'test_submission_statistics.py', 'verify_supplement.py',
                 'bfcl_fixed_replay.py', 'test_bfcl_fixed_replay.py', 'verify_mechanism_evidence.py',
                 'native_fc_results.py']:
        add('analysis/'+name,'analysis/'+name)
    for p in sorted((ROOT/'p4').glob('PREREGISTRATION*.md')):
        add(str(p.relative_to(ROOT)),p.relative_to(ROOT))
    add('p3/PREREG_AMENDMENT_20260903.md','p3/PREREG_AMENDMENT_20260903.md')
    # The public abstract is not an input to the anonymous main file.
    files.pop('paper/sections/abstract.tex',None)
    # These appendices now have ICLR-specific revisions; exclude unused public copies.
    files.pop('paper/sections/G_extra.tex',None)
    files.pop('paper/sections/H_limitations.tex',None)
    # Only the outcome fields used by the test are included, not task/model text.
    for filename in FILES.values():
        clean=[]
        for line in (ROOT/'runs/final'/filename).read_text().splitlines():
            r=json.loads(line); first=r['turns'][0]
            clean.append(dict(clean_index=r['clean_index'],first_ok=r['first_ok'],final_ok=r['final_ok'],
                turns=[dict(action=bool(first.get('action')),_fc_arg_keys=first.get('_fc_arg_keys',[]))]))
        files['runs/final/'+filename]=(''.join(json.dumps(r)+'\n' for r in clean)).encode()
    audit=json.loads((ROOT/'p3/results/formal_20260916_a/audit.json').read_text())
    files['p3/aggregate_evidence.json']=(json.dumps({k:audit[k] for k in
        ['final_lineage','evaluation','events','training_diagnostics','file_manifest']},indent=2)+'\n').encode()
    config=dict(model='Qwen2.5-Coder-7B-Instruct',framework='verl 0.9.0',rollout='vLLM 0.27.1',
        algorithm='GRPO',seed=0,steps=150,learning_rate=1e-6,lora_rank=32,lora_alpha=32,
        prompts_per_step=16,rollouts_per_prompt=8,temperature=1.0,top_p=1.0,top_k=-1,
        prompt_tokens=2048,response_tokens=6144,max_assistant_turns=5,max_parallel_calls=1,
        max_tool_response_length=1200,broken_parser='hermes',repaired_parser='qwen2_5_coder',
        evaluation=dict(multi_items=542,repair_items=454,temperature=0,max_turns=4,tokens_per_turn=1024,
            protocol='shared parser-independent generated-code execution with automatic test feedback',
            steps=[0,30,60,90,120,150]),
        decontamination=dict(text_ngrams=[8,5],code_ngrams=[10,6],minimum_grams=5,
            keep_when_max_containment_below=0.10))
    files['p3/configuration_summary.json']=(json.dumps(config,indent=2)+'\n').encode()
    for name in ('evaluation_outcomes.jsonl','events_compact.jsonl.gz','lineage.json','expected.json'):
        add('p3/compact/'+name,Path('submission_packages/p3_compact')/name)
    assert 'p3/compact/events_compact.jsonl.gz' in files, 'Run build_p3_supplement.py first'
    add('VERIFICATION_SCOPE.md','analysis/VERIFICATION_SCOPE.md')
    for name in ['outcomes.json','statistics.json','configurations.json']:
        add('native_fc/'+name,Path('submission_packages/native_fc')/name)
    add('mechanism_evidence/README.md','analysis/MECHANISM_EVIDENCE.md')
    for p in sorted((OUT/'mechanism_evidence').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            add(str(p.relative_to(OUT)),p.relative_to(ROOT))
    files['README.md']=b'''# Anonymous submission materials

Compile paper/iclr2027/main.tex using pdfLaTeX (latexmk -pdf).
The included PDF is the local candidate; the submission portal has not been updated.

Run from this directory:
python3 analysis/submission_statistics.py --output /tmp/recomputed_statistics.json
Compare that output with analysis/submission_statistics.json.
python3 analysis/verify_supplement.py
python3 analysis/verify_mechanism_evidence.py
python3 analysis/bfcl_fixed_replay.py --bundle mechanism_evidence/bfcl
python3 analysis/native_fc_results.py --outcomes native_fc/outcomes.json --out /tmp/native-fc-recheck

Contents: anonymous paper and sources, exact statistical code and outputs, reduced
outcome inputs for the 15 Llama tests, P3 reduced per-item outcomes and compact event records,
P3 aggregate checkpoint/event/log diagnostics,
and a configuration summary. Preregistration documents are retained with their
original predictions, amendments and commit identifiers. Historical other tests are recomputed from explicitly
listed discordant counts, not re-audited here from their raw trajectories.

Data boundary: reduced Llama records omit prompts and outputs but preserve all
fields read by the statistical script. The raw-content mechanism subset includes
BFCL response/schema/scoring records, all tau-bench task trajectories and twenty
deterministically sampled RL trajectories. See mechanism_evidence/README.md.
The complete P3 raw rollout archive, unreduced evaluation rows,
original source manifests and heavy checkpoints are NOT included. This supplement
reproduces statistics, recorded event totals and the accepted resume lineage,
not training or the complete raw-event/checkpoint audit. Only supplied raw-content
subsets support independent inspection of emission labels.
See VERIFICATION_SCOPE.md. Full records remain retained separately. No inference
or additional GPU work was run to assemble this package.

Native FC adds 1,626 reduced per-item outcomes, including timeout statuses,
anonymized frozen configuration records (including exact schemas, package versions
and source hashes), the 540-item descriptive sensitivity and final-submission
rescue alternative, and
nine post-hoc exploratory paired contrasts. Full native FC generations, original
tests and GPU checkpoints are not in this reduction; it supports statistical
recomputation, not raw-text reclassification or environment re-execution.

The historical 27-test multiplicity rule is retrospective, not preregistered. Endpoint
intervals cover item-pair uncertainty under independence, not seed or retest noise.
'''
    forbidden=re.compile(rb'wangwenbo|wenbwang|Wenbo Wang|Di Sang|3120265429|nebula-1999|/Users/|@bit\.edu|@my\.cityu',re.I)
    for name,data in files.items():
        if name.endswith(('.tex','.md','.py','.json','.jsonl','.txt','.bib','.bbl')):
            assert not forbidden.search(data),f'Identity marker in {name}'
    out=OUT/'iclr_anonymous_submission.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()): z.writestr(name,data)
    print(f'{out}: {len(files)} files, {out.stat().st_size} bytes; text identity scan passed')


if __name__=='__main__':main()
