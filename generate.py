"""Generate code from your own instruction with a trained checkpoint.

python generate.py --checkpoint runs/RUN/checkpoint_epoch_0030.pt "Write solve(numbers) that will count the values in numbers that are at most 4."
python generate.py --checkpoint runs/RUN/checkpoint_epoch_0030.pt --trace "..."
python generate.py --checkpoint runs/RUN/checkpoint_epoch_0030.pt          # interactive

Pipeline shown step by step with --trace:
  instruction -> lowercase word tokens -> ids (+<EOS>; unknown words -> <UNK>)
  -> encoder -> decoder starts from <SOS>, greedily picks the most probable
  next code token (PAD/SOS/UNK excluded) until <EOS> or 64 tokens
  -> code tokens -> Python source (indentation rebuilt from INDENT/DEDENT).
The generated code is printed, NEVER executed.
"""
import argparse

import torch
from model import Seq2Seq, PAD_ID, UNK_ID, SOS_ID, EOS_ID
from preprocess import tokenize_instruction, encode, detokenize_code


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = Seq2Seq(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['vocabularies']


@torch.no_grad()
def generate(model, vocab, instruction, max_new_tokens=64, trace=False):
    words = tokenize_instruction(instruction)
    source_tokens = words + ['<EOS>']
    ids = encode(words, vocab['source_token_to_id']) + [EOS_ID]
    if trace:
        unknown = [w for w in words if w not in vocab['source_token_to_id']]
        print('Source tokens:', source_tokens)
        print('Source ids:   ', ids)
        if unknown:
            print('Unknown words (mapped to <UNK>):', unknown)
    source = torch.tensor([ids])
    lengths = torch.tensor([len(ids)])
    state, memory = model._encode(source, lengths)
    previous = torch.tensor([[SOS_ID]])
    output = []
    id_to_token = vocab['target_id_to_token']
    for step in range(max_new_tokens):
        logits, state, weights = model._decode(previous, state, memory)
        scores = logits[0, -1].clone()
        scores[[PAD_ID, UNK_ID, SOS_ID]] = -torch.inf
        probabilities = torch.softmax(scores, dim=-1)
        best = int(probabilities.argmax())
        if trace:
            top = torch.topk(probabilities, 3)
            alternatives = ', '.join(f'{id_to_token[int(i)]} {float(p):.3f}'
                                     for p, i in zip(top.values, top.indices))
            line = f'  step {step:2d}: {id_to_token[best]:<14} | top-3: {alternatives}'
            if weights is not None:
                w = weights[0, -1]
                k = int(w.argmax())
                line += f' | most attended: {source_tokens[k]!r} ({float(w[k]):.2f})'
            print(line)
        if best == EOS_ID:
            break
        output.append(id_to_token[best])
        previous = torch.tensor([[best]])
    else:
        print('Warning: reached the length limit without <EOS>.')
    try:
        return detokenize_code(output), output
    except (ValueError, IndexError, IndentationError, SyntaxError) as error:
        return f'<could not rebuild code: {error}>', output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--trace', action='store_true', help='Print every decoding step.')
    parser.add_argument('instructions', nargs='*')
    args = parser.parse_args()
    model, vocab = load(args.checkpoint)
    variant = ('bidirectional ' if model.bidirectional else '') + ('attention' if model.attention else 'baseline')
    print(f'Loaded {variant} model from {args.checkpoint}\n')
    instructions = args.instructions
    interactive = not instructions
    while True:
        if interactive:
            try:
                instruction = input('Instruction (empty line to quit): ').strip()
            except EOFError:
                break
            if not instruction:
                break
        else:
            if not instructions:
                break
            instruction = instructions.pop(0)
        print(f'Instruction: {instruction}')
        code, _ = generate(model, vocab, instruction, trace=args.trace)
        print('Generated code:\n' + code)


if __name__ == '__main__':
    main()
