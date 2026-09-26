#!/usr/bin/env python3
"""Synthetic integration check, never a model-initiation or benchmark result."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_fc_eval import HFPolicy, evaluate_task
from sandbox import run_many


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--allow-gpu', action='store_true')
    a = p.parse_args()
    if not a.allow_gpu:
        p.error('Explicit GPU permission required')
    if a.out.exists():
        raise FileExistsError(a.out)
    policy = HFPolicy(a.base, None, max_tokens=128)
    generated = []
    fixture = json.dumps({'name': 'run_tests', 'arguments': {'code': 'def f():\n    return 1'}})

    def generate(messages):
        if len(messages) == 2:
            return fixture, 'injected_fixture_not_model_generation'
        assert messages[-1]['role'] == 'tool'
        rendered = policy.tokenizer.apply_chat_template(
            messages, tools=__import__('native_fc_eval').TOOLS,
            tokenize=False, add_generation_prompt=True)
        assert '<tool_response>' in rendered
        raw, finish = policy(messages)
        assert raw.strip(), 'Empty real-model response to tool feedback'
        generated.append({'raw': raw, 'finish': finish})
        return raw, finish

    def test(code):
        r = run_many([(code, 'assert f() == 1\n')], workers=1,
                     mode='script', timeout=30)[0]
        return dict(passed=r.passed, total=r.total, all_passed=r.all_passed,
                    status=r.status, stderr=r.stderr)

    row = evaluate_task('synthetic-feedback-fixture', 'Implement f returning 1.',
                        'You are a Python assistant.', generate, test, max_turns=2)
    assert row['events'][0]['executed'] and row['events'][0]['result']['all_passed']
    assert len(generated) == 1
    result = dict(status='PASS', synthetic=True, model_initiation_measured=False,
                  identity=policy.identity, trajectory=row, real_generations=generated)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('GPU_FEEDBACK_FIXTURE_PASS: injected call, real execution, real GPU continuation')


if __name__ == '__main__':
    main()
