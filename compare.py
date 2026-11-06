"""Compare evaluated runs (baseline vs attention) side by side. No training.

python compare.py runs/baseline_seed42_* runs/attention_seed42_* [more runs...]

Each run folder needs config.json and at least one evaluation_*/summary.json
(produced by evaluate.py). The most recent evaluation in each run is used.
Writes comparison.md and comparison.json in the current directory (or --out).
Test results are reported, never used to choose settings.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HEADLINE = ['all_tests_pass_rate', 'exact_token_match_rate', 'syntax_validity_rate',
            'operation_correct_rate', 'relation_correct_rate', 'threshold_correct_rate',
            'teacher_forced_token_accuracy', 'perplexity']
NOT_PERCENT = {'perplexity'}


def load_run(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    evaluations = sorted(path.glob('evaluation_*/summary.json'))
    if not evaluations:
        raise SystemExit(f'No evaluation found in {path}; run evaluate.py first.')
    evaluation = json.loads(evaluations[-1].read_text(encoding='utf-8'))
    train_summary = path / 'summary.json'
    best_epoch = None
    if train_summary.exists():
        best_epoch = json.loads(train_summary.read_text(encoding='utf-8')).get('best_epoch')
    variant = evaluation.get('variant') or ('attention' if config['model_config'].get('attention') else 'baseline')
    return {'run': path.name, 'variant': variant, 'seed': config.get('seed'),
            'device': config.get('device'), 'best_epoch': best_epoch,
            'parameters': evaluation.get('trainable_parameters'),
            'splits': evaluation['splits']}


def pct(x):
    return '–' if x is None else f'{100*x:.1f}%'


def fmt(key, x):
    if x is None:
        return '–'
    return f'{x:.4f}' if key in NOT_PERCENT else pct(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()
    selected = []
    for r in args.runs:
        mode = json.loads((Path(r) / 'config.json').read_text(encoding='utf-8')).get('mode')
        if mode == 'overfit':
            print(f'Skipping overfit diagnostic run: {r}')
            continue
        selected.append(r)
    runs = [load_run(r) for r in selected]
    out = Path(args.out)
    lines = ['# Baseline vs attention: controlled comparison', '',
             'Same data, splits, vocabulary, optimizer, LR, batch size, max epochs, '
             'patience, teacher forcing, clipping and validation-based checkpoint selection. '
             'Only the decoder differs.', '']
    lines += ['## Runs', '', '| Run | Variant | Seed | Device | Best epoch | Parameters |',
              '|---|---|---:|---|---:|---:|']
    for r in runs:
        lines.append(f"| {r['run']} | {r['variant']} | {r['seed']} | {r['device']} | "
                     f"{r['best_epoch']} | {r['parameters']:,} |" if r['parameters'] else
                     f"| {r['run']} | {r['variant']} | {r['seed']} | {r['device']} | {r['best_epoch']} | – |")
    for split in ['test_wording', 'test_composition']:
        lines += ['', f'## {split}', '', '| Run | Variant | Seed | ' +
                  ' | '.join(k.replace('_rate', '') for k in HEADLINE) + ' |',
                  '|---|---|---:|' + '---:|' * len(HEADLINE)]
        for r in runs:
            m = r['splits'][split]
            lines.append(f"| {r['run']} | {r['variant']} | {r['seed']} | " +
                         ' | '.join(fmt(k, m.get(k)) for k in HEADLINE) + ' |')
        # Mean and range across seeds per variant.
        grouped = defaultdict(list)
        for r in runs:
            grouped[r['variant']].append(r['splits'][split])
        if any(len(v) > 1 for v in grouped.values()):
            lines += ['', f'Mean [min–max] across seeds, {split}:', '',
                      '| Variant | Runs | ' + ' | '.join(k.replace('_rate', '') for k in HEADLINE) + ' |',
                      '|---|---:|' + '---:|' * len(HEADLINE)]
            for variant, ms in grouped.items():
                cells = []
                for k in HEADLINE:
                    vals = [m[k] for m in ms if k in m]
                    cells.append(f'{fmt(k, statistics.mean(vals))} [{fmt(k, min(vals))}–{fmt(k, max(vals))}]'
                                 if vals else '–')
                lines.append(f'| {variant} | {len(ms)} | ' + ' | '.join(cells) + ' |')
    structures = sorted(runs[0]['splits']['test_composition']['by_structure'])
    by_variant = defaultdict(list)
    for r in runs:
        by_variant[r['variant']].append(r['splits']['test_composition']['by_structure'])
    lines += ['', '## test_composition by held-out structure: mean [min–max] across seeds', '',
              '| Variant | Runs | ' + ' | '.join(f'{s} pass | {s} relation' for s in structures) + ' |',
              '|---|---:|' + '---:|' * (2 * len(structures))]
    for variant, ms in by_variant.items():
        cells = []
        for s in structures:
            for k in ['all_tests_pass_rate', 'relation_correct_rate']:
                vals = [m[s][k] for m in ms]
                cells.append(f'{pct(statistics.mean(vals))} [{pct(min(vals))}–{pct(max(vals))}]')
        lines.append(f'| {variant} | {len(ms)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## test_composition by held-out structure, per run', '',
              '| Run | Variant | Seed | Structure | All tests pass | Operation | Relation | Threshold | Generated relations |',
              '|---|---|---:|---|---:|---:|---:|---:|---|']
    for r in runs:
        for structure, m in r['splits']['test_composition']['by_structure'].items():
            confusion = next(iter(m.get('relation_confusion', {}).values()), {})
            generated = ', '.join(f'{k}:{v}' for k, v in sorted(confusion.items(), key=lambda kv: -kv[1]))
            lines.append(f"| {r['run']} | {r['variant']} | {r['seed']} | {structure} | "
                         f"{pct(m['all_tests_pass_rate'])} | {pct(m.get('operation_correct_rate'))} | "
                         f"{pct(m.get('relation_correct_rate'))} | {pct(m.get('threshold_correct_rate'))} | {generated} |")
    lines += ['', 'Behavioral checks use the bounded AST interpreter in evaluate.py, '
              'not unrestricted execution. Composition test has 96 examples / 48 tasks: '
              'one held-out structure = 32 examples, so each example is ~3 points on that row.']
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'comparison.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
    print('\n'.join(lines))
    print(f"\nSaved {out/'comparison.md'} and {out/'comparison.json'}")


if __name__ == '__main__':
    main()
