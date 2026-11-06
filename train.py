"""Train baseline or attention model using train and validation_wording ONLY.

python train.py --overfit 16 --epochs 400              # tiny-set diagnostic first
python train.py                                        # baseline, max 30 epochs
python train.py --overfit 16 --epochs 400 --attention  # same diagnostic, attention
python train.py --attention                            # attention, SAME settings
python train.py --bidirectional [--attention]          # bidirectional encoder variants

--attention changes ONLY the decoder (see model.py). Data, splits, vocabulary,
optimizer, LR, batch size, epochs, patience, seed, teacher forcing, clipping and
checkpoint selection are identical, so the two runs form a controlled comparison.

Needs model.py, preprocess.py and processed/ beside this file.
No generated code is executed. Test sets are never opened.
Each invocation creates a new timestamped runs/ directory.
Improved checkpoints use unique epoch filenames: no replacement/rename operation.
--init-checkpoint PATH recovers model weights, with a NEW optimizer and epoch count.
It is a warm restart, not an exact continuation of interrupted training.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Seq2Seq, collate_batch, PAD_ID, EOS_ID
from preprocess import detokenize_code


def load_rows(path):
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def make_loader(rows, batch_size, shuffle=False, seed=42):
    # num_workers=0 avoids multiprocessing setup issues on Windows.
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def loss_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')
    total_loss = total_tokens = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            source = batch['source_ids'].to(device)
            inputs = batch['decoder_input_ids'].to(device)
            targets = batch['decoder_target_ids'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(source, batch['source_lengths'], inputs)
            loss_sum = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            token_count = targets.ne(PAD_ID).sum().item()
            if not torch.isfinite(loss_sum).item():
                raise RuntimeError('Non-finite loss; training stopped.')
            if training:
                (loss_sum / token_count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                         error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss_sum.detach().item()
            total_tokens += token_count
    return total_loss / total_tokens


@torch.no_grad()
def generation_check(model, rows, device, vocab, batch_size=32):
    """Exact token match is a diagnostic, NOT functional correctness."""
    exact = ended = 0
    examples = []
    id_to_token = vocab['target_id_to_token']
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        batch = collate_batch(chunk)
        outputs, flags = model.generate(batch['source_ids'].to(device),
                                        batch['source_lengths'], max_new_tokens=64)
        for row, output, eos in zip(chunk, outputs, flags):
            reference = row['decoder_target_ids']
            assert reference[-1] == EOS_ID
            match = eos and output == reference[:-1]
            exact += int(match)
            ended += int(eos)
            if len(examples) < 8:
                tokens = [id_to_token[i] for i in output]
                try:
                    code = detokenize_code(tokens)
                except (ValueError, IndexError, IndentationError, SyntaxError) as error:
                    code = None
                examples.append({'id': row['id'], 'exact_token_match': bool(match),
                    'ended_with_eos': eos, 'generated_tokens': tokens,
                    'generated_code': code,
                    'reference_code': detokenize_code([id_to_token[i] for i in reference[:-1]])})
    return {'exact_token_match_rate': exact / len(rows),
            'eos_rate': ended / len(rows), 'examples_checked': len(rows),
            'sample_outputs': examples}


def save_checkpoint(run, epoch, model, model_config, vocab, loss, metric, config):
    """Never overwrite or rename a checkpoint; record success only after close."""
    path = run / f'checkpoint_epoch_{epoch:04d}.pt'
    with path.open('xb') as handle:
        torch.save({'model_state_dict': model.state_dict(),
                    'model_config': model_config, 'vocabularies': vocab,
                    'epoch': epoch, 'selection_loss': loss,
                    'selection_metric': metric, 'config': config}, handle)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overfit', type=int, default=0,
                        help='Fit N distinct training tasks; NOT a validation experiment.')
    parser.add_argument('--attention', action='store_true',
                        help='Use the Luong-attention decoder; everything else unchanged.')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Bidirectional encoder (same total hidden size); everything else unchanged.')
    parser.add_argument('--init-checkpoint', default=None,
                        help='Warm restart from saved weights; optimizer and epoch count reset.')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.overfit < 0:
        parser.error('Epochs, batch size, patience and LR must be positive; overfit >= 0.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    # Seeds reduce variability; exact cross-device reproducibility is not guaranteed.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    root = Path(__file__).resolve().parent
    processed = root / 'processed'
    vocab_path = processed / 'vocabularies.json'
    if not vocab_path.exists():
        raise SystemExit('Run preprocess.py first.')
    vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
    for key in ['source_token_to_id','target_token_to_id']:
        assert [vocab[key][t] for t in ['<PAD>','<UNK>','<SOS>','<EOS>']] == [0,1,2,3]
    train_rows = load_rows(processed / 'train.jsonl')
    variant = ('bi_' if args.bidirectional else '') + ('attention' if args.attention else 'baseline')
    mode = 'overfit' if args.overfit else variant
    used_files = [processed/'train.jsonl', vocab_path]
    if args.overfit:
        selected, seen = [], set()
        for row in train_rows:
            if row['task_id'] not in seen:
                selected.append(row)
                seen.add(row['task_id'])
            if len(selected) == args.overfit:
                break
        if len(selected) < args.overfit:
            parser.error('Requested more distinct tasks than available.')
        train_rows = selected
        evaluation_rows = selected
        evaluation_label = 'tiny_train_loss'
    else:
        evaluation_rows = load_rows(processed / 'validation_wording.jsonl')
        used_files.append(processed/'validation_wording.jsonl')
        evaluation_label = 'validation_loss'
    run_name = f'overfit_{variant}' if args.overfit else variant
    run = root/'runs'/f'{run_name}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = {'source_vocab_size':len(vocab['source_token_to_id']),
                    'target_vocab_size':len(vocab['target_token_to_id']),
                    'embedding_dim':128,'hidden_dim':256,
                    'attention':args.attention,'bidirectional':args.bidirectional}
    model = Seq2Seq(**model_config).to(device)
    recovered = None
    if args.init_checkpoint:
        recovered = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        # Checkpoints saved before the attention option existed are baselines.
        recovered_config = {'attention': False, 'bidirectional': False, **recovered['model_config']}
        if recovered_config != model_config or recovered['vocabularies'] != vocab:
            raise ValueError('Checkpoint architecture or vocabularies do not match this dataset.')
        model.load_state_dict(recovered['model_state_dict'])
        print(f"Recovered weights from old epoch {recovered.get('epoch', '?')}. "
              'Starting a new optimizer and epoch count.', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(train_rows, args.batch_size, True, args.seed)
    evaluation_loader = make_loader(evaluation_rows, args.batch_size)
    config = {**vars(args), 'mode':mode,'variant':variant,'model_config':model_config,
              'trainable_parameters':sum(p.numel() for p in model.parameters()),
              'teacher_forcing_ratio':1.0,'gradient_clip_norm':1.0,
              'device':str(device),'torch_version':str(torch.__version__),
              'train_examples':len(train_rows),'evaluation_examples':len(evaluation_rows),
              'evaluation_label':evaluation_label,
              'file_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in used_files}}
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Device: {device} | Mode: {mode} | Variant: {variant} | '
          f"Parameters: {config['trainable_parameters']:,} | Run: {run}", flush=True)
    if args.overfit:
        print('Diagnostic only: fitting and checking the SAME tiny training subset.', flush=True)
    else:
        print('Checkpoint selection uses validation_wording; test sets remain untouched.', flush=True)
    initial = loss_epoch(model, evaluation_loader, device)
    print(f'Initial {evaluation_label}: {initial:.4f}', flush=True)
    best, best_epoch, stale = initial, 0, 0
    best_path = save_checkpoint(run, 0, model, model_config, vocab,
                                best, evaluation_label, config)
    started = time.perf_counter()
    with (run/'history.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=['epoch','train_loss',evaluation_label,'epoch_seconds'])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            epoch_start = time.perf_counter()
            train_loss = loss_epoch(model, train_loader, device, optimizer)
            evaluation_loss = loss_epoch(model, evaluation_loader, device)
            elapsed = time.perf_counter()-epoch_start
            writer.writerow({'epoch':epoch,'train_loss':train_loss,
                             evaluation_label:evaluation_loss,'epoch_seconds':elapsed})
            handle.flush()
            improved = evaluation_loss < best
            if improved:
                best, best_epoch, stale = evaluation_loss, epoch, 0
                best_path = save_checkpoint(run, epoch, model, model_config, vocab,
                                            best, evaluation_label, config)
            else:
                stale += 1
            print(f'Epoch {epoch:03d} | train={train_loss:.4f} | '
                  f'{evaluation_label}={evaluation_loss:.4f} | {elapsed:.1f}s'
                  + (' | saved best' if improved else ''),flush=True)
            if not args.overfit and stale >= args.patience:
                print(f'Early stopping after {args.patience} epochs without improvement.',flush=True)
                break
    # Reload our own checkpoint and generate with no teacher forcing.
    checkpoint = torch.load(best_path,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    generation = generation_check(model, evaluation_rows, device, vocab, args.batch_size)
    summary = {'mode':mode,'variant':variant,'best_epoch':best_epoch,'best_checkpoint':best_path.name,
               'selection_metric':evaluation_label,
               'initial_selection_loss':initial,'best_selection_loss':best,
               'total_seconds':time.perf_counter()-started,
               'generation_dataset':'same tiny training subset' if args.overfit else 'validation_wording',
               **generation}
    (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'\nBest epoch: {best_epoch} | {evaluation_label}: {best:.4f}')
    print(f"Greedy exact token match ({summary['generation_dataset']}): "
          f"{generation['exact_token_match_rate']:.1%}")
    print('This is token matching, not execution-based correctness.')
    print(f'Best checkpoint: {best_path.name}')
    print('Saved: epoch checkpoints, history.csv, config.json, summary.json')
    print(f'Run folder: {run}')


if __name__ == '__main__':
    main()
