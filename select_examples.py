"""Pick qualitative examples (successes and failures) from saved predictions.

python select_examples.py --out results_attention_comparison/examples.md

Deterministic: for each case below, the FIRST matching prediction (file order)
in that run's most recent evaluation is shown. Nothing is re-generated here.
Cases are chosen to illustrate failure TYPES found in the aggregate metrics;
they are illustrations, not additional evidence.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CASES = [
    # (heading, run glob, split, predicate, comment)
    ('Success: held-out filter + "!=" (attention, seed 42)',
     'attention_seed42_*', 'test_composition',
     lambda r: r['structure'] == 'filter_ne' and r['all_tests_passed'],
     'This combination never appears in training. Operation, relation and threshold are all correct.'),
    ('Success: held-out count + "<=" (attention, seed 42)',
     'attention_seed42_*', 'test_composition',
     lambda r: r['structure'] == 'count_le' and r['all_tests_passed'] and r['requested_threshold'].startswith('-'),
     'A negative threshold is spread over two tokens ("-", "5") and is still copied correctly.'),
    ('Success: held-out sum + ">" (attention, seed 42)',
     'attention_seed42_*', 'test_composition',
     lambda r: r['structure'] == 'sum_gt' and r['all_tests_passed'],
     ''),
    ('Success: unseen wording wrapper (attention, seed 42)',
     'attention_seed42_*', 'test_wording',
     lambda r: r['family'] == 'lists' and r['structure'].startswith('count') and r['all_tests_passed'],
     '"using" and "signature" are unknown words (<UNK>) for the model; the output is still correct.'),
    ('Failure: baseline reverses "not equal" (baseline, seed 42)',
     'baseline_seed42_*', 'test_composition',
     lambda r: r['structure'] == 'filter_ne' and r.get('generated_relation') == 'eq',
     'Operation and threshold are right; only the relation is wrong. `filter` was only ever seen with '
     'eq/gt/ge/lt/le in training, and the decoder falls back to one of them.'),
    ('Failure: bidirectional encoder without attention (seed 7)',
     'bi_baseline_seed7_*', 'test_composition',
     lambda r: r['structure'] == 'sum_gt' and not r['all_tests_passed'],
     'Again a comparison already seen with this operation (`sum` was trained with `!=`) replaces the requested one. '
     'Reading the sentence in both directions does not help when the decoder sees it only through the final state.'),
    ('Failure: "count" becomes "sum" under unseen wording (baseline, seed 123)',
     'baseline_seed123_*', 'test_wording',
     lambda r: r['structure'].startswith('count') and r.get('generated_operation') == 'sum',
     'With the unknown words "using" and "signature", the decoder confuses the two operations that both start with `sum(`.'),
    ('Failure: the same confusion with attention in one run (bidirectional + attention, seed 2)',
     'bi_attention_seed2_*', 'test_wording',
     lambda r: r['structure'].startswith('count') and r.get('generated_operation') == 'sum',
     'The only attention run with this error (26% of its wording examples). Composition is still 100% in this run.'),
    ('Failure: tiny family (attention, seed 2)',
     'attention_seed2_*', 'test_wording',
     lambda r: not r['all_tests_passed'],
     'Arithmetic has only 32 training examples. The model writes a product instead of a difference; '
     'these are the only 2 wrong programs of this run out of 500.'),
]


def latest_predictions(run_glob, split):
    runs = sorted((ROOT / 'runs').glob(run_glob))
    if not runs:
        raise SystemExit(f'No run matches {run_glob}')
    evaluation = sorted(runs[-1].glob('evaluation_*'))[-1]
    path = evaluation / f'{split}_predictions.jsonl'
    return runs[-1].name, [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def block(code):
    return '```python\n' + (code or '<no reconstructable code>').rstrip() + '\n```'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results_attention_comparison/examples.md')
    args = parser.parse_args()
    lines = ['# Generated examples', '',
             'Each example is the first matching prediction in the saved evaluation output '
             '(see select_examples.py). Generated code is shown as rebuilt from tokens; spacing '
             'differs from the reference but is not counted as an error. Correctness = all '
             'behavioural tests pass.', '']
    for heading, run_glob, split, predicate, comment in CASES:
        run, rows = latest_predictions(run_glob, split)
        match = next((r for r in rows if predicate(r)), None)
        lines += [f'## {heading}', '']
        if match is None:
            lines += ['_No matching example in this run._', '']
            continue
        lines += [f'Run `{run}`, split `{split}`, example `{match["id"]}`', '',
                  f'**Input:** {match["instruction"]}', '',
                  '**Reference:**', block(match['reference_code']), '',
                  '**Generated:**', block(match['generated_code']), '',
                  f'**Result:** {"all tests pass" if match["all_tests_passed"] else match["failure"]}']
        if 'requested_relation' in match:
            lines.append(f'(requested: {match["requested_operation"]} / {match["requested_relation"]} / '
                         f'{match["requested_threshold"]}; generated: {match["generated_operation"]} / '
                         f'{match["generated_relation"]} / {match["generated_threshold"]})')
        if comment:
            lines += ['', comment]
        lines.append('')
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
