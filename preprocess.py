"""Prepare the synthetic Python-generation dataset (Python 3.10+, no dependencies).

Place beside generate_dataset.py and run: python preprocess.py
Reads data/*.jsonl; writes processed/*.jsonl, vocabularies.json and report.json.
Vocabularies are built ONLY from train.jsonl. No code is executed.
Sequences are variable-length; batch padding belongs in the training DataLoader.
No truncation is applied: this preserves complete functions.
"""
import ast
import io
import json
import re
import token
import tokenize
from collections import Counter
from pathlib import Path

SPECIALS = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
SPLITS = ['train', 'validation_wording', 'test_wording', 'test_composition']
STRUCTURAL = {
    tokenize.NEWLINE: '<NEWLINE>',
    tokenize.INDENT: '<INDENT>',
    tokenize.DEDENT: '<DEDENT>',
    tokenize.NL: '<NL>',
}
REVERSE_STRUCTURAL = {v: k for k, v in STRUCTURAL.items()}


def tokenize_instruction(text):
    """Lowercase English; retain identifiers, digits and punctuation as tokens.

    Source normalization is safe for this pilot's fixed lowercase signatures.
    Do not reuse blindly for instructions containing case-sensitive literals.
    """
    return re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\w\s]', text.lower())


def tokenize_code(code):
    """Token types distinguish identifiers, numbers, strings and operators.

    Case and literal spelling are preserved. Indentation is normalized on decode.
    Comments are preserved, though this pilot has none.
    """
    result = []
    for item in tokenize.generate_tokens(io.StringIO(code).readline):
        if item.type == tokenize.ENDMARKER:
            continue
        if item.type in STRUCTURAL:
            result.append(STRUCTURAL[item.type])
        else:
            result.append(f'{token.tok_name[item.type]}:{item.string}')
    return result


def detokenize_code(tokens):
    """Reconstruct code from lexical tokens; formatting may differ from source.

    Unknown or padding tokens in a generated code sequence are errors. SOS/EOS
    should be stripped by the generation routine, not silently discarded here.
    """
    pairs = []
    depth = 0
    for item in tokens:
        if item in REVERSE_STRUCTURAL:
            kind = REVERSE_STRUCTURAL[item]
            if kind == tokenize.INDENT:
                depth += 1
                value = '    ' * depth
            elif kind == tokenize.DEDENT:
                depth -= 1
                if depth < 0:
                    raise ValueError('Unmatched DEDENT')
                value = ''
            else:
                value = '\n'
        else:
            name, separator, value = item.partition(':')
            if not separator or name not in token.__dict__:
                raise ValueError(f'Invalid code token: {item!r}')
            kind = token.__dict__[name]
            if not isinstance(kind, int):
                raise ValueError(f'Invalid token type: {name}')
        pairs.append((kind, value))
    if depth:
        raise ValueError('Unclosed indentation block')
    return tokenize.untokenize(pairs)


def build_vocabulary(sequences):
    counts = Counter(t for sequence in sequences for t in sequence)
    # IDs 0-3 are reserved. Sorting makes remaining IDs deterministic.
    ordinary = sorted(t for t in counts if t not in SPECIALS)
    return {t: i for i, t in enumerate(SPECIALS + ordinary)}


def encode(tokens, vocabulary):
    return [vocabulary.get(t, vocabulary['<UNK>']) for t in tokens]


def write_json(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    root = Path(__file__).resolve().parent
    loaded = {}
    for split in SPLITS:
        path = root / 'data' / f'{split}.jsonl'
        if not path.exists():
            raise SystemExit(f'Missing {path}. Run python generate_dataset.py first.')
        with path.open(encoding='utf-8') as handle:
            loaded[split] = [json.loads(line) for line in handle if line.strip()]
    train = loaded['train']
    source_vocab = build_vocabulary(tokenize_instruction(r['instruction']) for r in train)
    target_vocab = build_vocabulary(tokenize_code(r['code']) for r in train)
    output = root / 'processed'
    output.mkdir(exist_ok=True)
    write_json(output / 'vocabularies.json', {
        'source_token_to_id': source_vocab,
        'target_token_to_id': target_vocab,
        'source_id_to_token': list(source_vocab),
        'target_id_to_token': list(target_vocab),
        'special_ids': {s: source_vocab[s] for s in SPECIALS},
        'version': 1,
    })
    report = {
        'source_vocabulary_size': len(source_vocab),
        'target_vocabulary_size': len(target_vocab),
        'vocabulary_source': 'train only',
        'padding': 'deferred to batch collation; PAD ID 0',
        'truncation': 'none',
        'notes': [
            'Source EOS is appended; target decoder input starts with SOS.',
            'Decoder labels are code tokens followed by EOS.',
            'Reference AST comparison verifies syntax/structure, not model quality.',
            'Wording evaluation intentionally shares reference programs with training.',
            'Unseen English words map to UNK; test vocabulary is never added.',
        ],
        'splits': {},
    }
    checked = set()
    for split, rows in loaded.items():
        source_unknown = Counter()
        target_unknown = Counter()
        source_total = target_total = 0
        max_source = max_target = 0
        with (output / f'{split}.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                src = tokenize_instruction(row['instruction'])
                tgt = tokenize_code(row['code'])
                if row['code'] not in checked:
                    restored = detokenize_code(tgt)
                    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(row['code'])), row['id']
                    assert tokenize_code(restored) == tgt, row['id']
                    checked.add(row['code'])
                source_unknown.update(t for t in src if t not in source_vocab)
                target_unknown.update(t for t in tgt if t not in target_vocab)
                source_total += len(src)
                target_total += len(tgt)
                source_ids = encode(src, source_vocab) + [source_vocab['<EOS>']]
                target_ids = encode(tgt, target_vocab)
                decoder_input = [target_vocab['<SOS>']] + target_ids
                decoder_labels = target_ids + [target_vocab['<EOS>']]
                assert decoder_input[1:] == decoder_labels[:-1]
                assert len(decoder_input) == len(decoder_labels)
                max_source = max(max_source, len(source_ids))
                max_target = max(max_target, len(decoder_labels))
                encoded = {
                    'id': row['id'], 'task_id': row['task_id'],
                    'family': row['family'], 'structure': row['structure'],
                    'source_ids': source_ids,
                    'decoder_input_ids': decoder_input,
                    'decoder_target_ids': decoder_labels,
                    'source_length': len(source_ids),
                    'target_length': len(decoder_labels),
                }
                handle.write(json.dumps(encoded) + '\n')
        report['splits'][split] = {
            'examples': len(rows),
            'max_source_length_including_eos': max_source,
            'max_target_length_including_eos': max_target,
            'source_unknown_count': sum(source_unknown.values()),
            'source_unknown_rate': sum(source_unknown.values()) / max(source_total, 1),
            'source_unknown_tokens': dict(source_unknown.most_common()),
            'target_unknown_count': sum(target_unknown.values()),
            'target_unknown_rate': sum(target_unknown.values()) / max(target_total, 1),
            'target_unknown_tokens': dict(target_unknown.most_common()),
        }
        if split == 'train':
            assert not source_unknown and not target_unknown
    report['distinct_reference_roundtrips_passed'] = len(checked)
    write_json(output / 'report.json', report)
    # A single readable example for inspecting tokenization and target shifting.
    sample = train[0]
    src = tokenize_instruction(sample['instruction'])
    tgt = tokenize_code(sample['code'])
    write_json(output / 'example.json', {
        'instruction': sample['instruction'], 'reference_code': sample['code'],
        'source_tokens': src, 'code_tokens': tgt,
        'source_ids': encode(src, source_vocab) + [source_vocab['<EOS>']],
        'decoder_input_tokens': ['<SOS>'] + tgt,
        'decoder_target_tokens': tgt + ['<EOS>'],
        'reconstructed_code': detokenize_code(tgt),
    })
    print(json.dumps(report, indent=2))
    print(f'\nSaved processed data to: {output}')
    print('Open processed/example.json to inspect one tokenized example.')


if __name__ == '__main__':
    main()
