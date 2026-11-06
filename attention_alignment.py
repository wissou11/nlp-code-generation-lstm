"""Does attention look at the condition? Diagnostic for attention checkpoints.

python attention_alignment.py runs/attention_seed42_* runs/bi_attention_seed42_* ...

For each list-task example in test_wording and test_composition, greedy-decode
and inspect the attention row at two decoder steps:
  relation step   - when the relation operator (<, <=, ==, !=, >=, >) is emitted
  threshold step  - when the first threshold token is emitted
Reports the attention MASS on the matching source span (relation phrase, e.g.
"not equal to"; threshold tokens, e.g. "- 3") and how often the single most
attended token lies inside that span. Uniform attention would put roughly
span_length / source_length of the mass there; that baseline is printed too.
Nothing is trained or tuned; generated code is never executed.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from model import Seq2Seq, collate_batch, EOS_ID, SOS_ID
from preprocess import tokenize_instruction, encode
from evaluate import parse_condition


def spans(source_tokens):
    """(relation span, threshold span) as index ranges in instruction tokens."""
    are = max(i for i, t in enumerate(source_tokens) if t == 'are')
    k = are + 1
    while k < len(source_tokens) and not (source_tokens[k].isdigit() or source_tokens[k] == '-'):
        k += 1
    end = k
    while end < len(source_tokens) and (source_tokens[end].isdigit() or source_tokens[end] == '-'):
        end += 1
    return range(are + 1, k), range(k, end)


def analyse(run, root):
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(run / summary['best_checkpoint'], map_location='cpu', weights_only=True)
    vocab = checkpoint['vocabularies']
    model = Seq2Seq(**checkpoint['model_config'])
    if not model.attention:
        raise SystemExit(f'{run} is not an attention model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    result = {'run': run.name, 'seed': checkpoint['config']['seed'],
              'bidirectional': model.bidirectional, 'splits': {}}
    for split in ['test_wording', 'test_composition']:
        rows = [json.loads(l) for l in (root / 'data' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        rows = [r for r in rows if r['family'] == 'lists']
        stats = {'relation_mass': [], 'relation_top_in_span': [], 'relation_uniform': [],
                 'threshold_mass': [], 'threshold_top_in_span': [], 'threshold_uniform': []}
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            sources = [tokenize_instruction(r['instruction']) for r in chunk]
            batch = collate_batch([{'source_ids': encode(s, vocab['source_token_to_id']) + [EOS_ID],
                                    'decoder_input_ids': [SOS_ID], 'decoder_target_ids': [EOS_ID]}
                                   for s in sources])
            outputs, _, attention = model.generate(batch['source_ids'], batch['source_lengths'],
                                                   64, return_attention=True)
            for source, ids, rows_att in zip(sources, outputs, attention):
                parsed = parse_condition([vocab['target_id_to_token'][i] for i in ids])
                if parsed is None or parsed['relation_index'] + 1 >= len(rows_att):
                    continue
                relation_span, threshold_span = spans(source)
                length = len(source) + 1  # + EOS
                for name, step, span in [('relation', parsed['relation_index'], relation_span),
                                         ('threshold', parsed['relation_index'] + 1, threshold_span)]:
                    weights = rows_att[step]
                    top = max(range(len(weights)), key=weights.__getitem__)
                    stats[f'{name}_mass'].append(sum(weights[i] for i in span))
                    stats[f'{name}_top_in_span'].append(top in span)
                    stats[f'{name}_uniform'].append(len(span) / length)
        result['splits'][split] = {k: statistics.mean(v) for k, v in stats.items()} | {
            'examples': len(stats['relation_mass'])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', default='attention_alignment.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    results = [analyse(Path(r), root) for r in args.runs]
    print('| Run | Encoder | Split | Relation mass (uniform) | Relation top in span | '
          'Threshold mass (uniform) | Threshold top in span |')
    print('|---|---|---|---:|---:|---:|---:|')
    for r in results:
        for split, m in r['splits'].items():
            print(f"| {r['run']} | {'bi' if r['bidirectional'] else 'uni'} | {split} | "
                  f"{m['relation_mass']:.2f} ({m['relation_uniform']:.2f}) | {m['relation_top_in_span']:.0%} | "
                  f"{m['threshold_mass']:.2f} ({m['threshold_uniform']:.2f}) | {m['threshold_top_in_span']:.0%} |")
    Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
