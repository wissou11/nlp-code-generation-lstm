"""Encoder-decoder, PyTorch 2.7+. No pretrained weights.

Two variants share the same encoder, embeddings, LSTM sizes and greedy decoding:
  attention=False  baseline: decoder sees the source ONLY through the final
                   encoder state (h, c).
  attention=True   Bahdanau (additive) attention, as in the course: at every
                   decoder step, e_ij = v^T tanh(W h_j + U s_{i-1}) scores each
                   encoder state against the previous decoder state; softmax
                   weights give the context vector, which is concatenated with
                   the input embedding before the decoder LSTM. Padding positions
                   are masked out of the attention.
The attention variant has extra parameters (W, U, v and a wider LSTM input);
report the parameter counts next to any comparison.

Place next to preprocess.py. Run with the project's Python environment:
    python model.py
Either variant can use a bidirectional encoder (bidirectional=True): same
total hidden size (hidden_dim//2 per direction, concatenated), so every
encoder position summarizes the WHOLE instruction, not only its prefix.

This runs structural checks on a training batch for all four variants,
NOT full training.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


def collate_batch(rows):
    """Right-pad sequences to the longest in this batch. Lengths stay on CPU."""
    def pad(key):
        return pad_sequence([torch.tensor(r[key], dtype=torch.long) for r in rows],
                            batch_first=True, padding_value=PAD_ID)
    return {
        'source_ids': pad('source_ids'),
        'source_lengths': torch.tensor([len(r['source_ids']) for r in rows]),
        'decoder_input_ids': pad('decoder_input_ids'),
        'decoder_target_ids': pad('decoder_target_ids'),
    }


class Encoder(nn.Module):
    """Unidirectional by default. bidirectional=True uses hidden_dim//2 per
    direction and concatenates them, so outputs and final states keep size
    hidden_dim and the decoder is unchanged (no extra bridge layer)."""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, bidirectional=False):
        super().__init__()
        if bidirectional and hidden_dim % 2:
            raise ValueError('hidden_dim must be even for a bidirectional encoder')
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, source_ids, source_lengths, return_outputs=False):
        embedded = self.embedding(source_ids)
        # Packing ensures final states describe real input, not padding tokens.
        packed = pack_padded_sequence(embedded, source_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        if self.bidirectional:
            # (2, B, H/2) forward/backward finals -> (1, B, H) decoder state.
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1).unsqueeze(0)
            cell = torch.cat([cell[0], cell[1]], dim=-1).unsqueeze(0)
        if not return_outputs:
            return hidden, cell
        # Per-position outputs for attention; padded positions are zeros and
        # are additionally excluded by the attention mask.
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True,
                                         total_length=source_ids.size(1))
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, state):
        outputs, state = self.lstm(self.embedding(token_ids), state)
        # Return raw logits. CrossEntropyLoss applies log-softmax internally.
        return self.output(outputs), state


class AttentionDecoder(nn.Module):
    """LSTM decoder with Bahdanau (additive) attention, as derived in the course.

    At decoder step i, with previous decoder hidden state s_{i-1} and encoder
    hidden states h_1 ... h_n:
        e_ij    = v^T tanh(W h_j + U s_{i-1})          alignment scores
        a_ij    = softmax_j(e_ij)  over real (non-PAD) source positions
        c_i     = sum_j a_ij h_j                         context vector
        x_i     = [embedding(y_{i-1}) ; c_i]             merged decoder input
        s_i     = LSTM(x_i, s_{i-1})
        P(y_i)  = softmax(W_o s_i)
    Course pseudo-code: score = FC(tanh(FC(EO) + FC(H))); attention weights =
    softmax(score); context = sum(weights * EO); merged = concat(embedding, context).
    Because each step's scores depend on the previous decoder state, steps are
    computed one after the other (also under teacher forcing).
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)   # applied to encoder states h_j
        self.U = nn.Linear(hidden_dim, hidden_dim)               # applied to decoder state s_{i-1}
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, state, encoder_outputs, source_mask):
        """Return logits (B,T,V), new state, attention weights (B,T,S)."""
        embedded = self.embedding(token_ids)
        keys = self.W(encoder_outputs)                            # W h_j, computed once
        hidden, cell = state
        outputs, all_weights = [], []
        for t in range(token_ids.size(1)):
            s_prev = hidden[-1]                                   # s_{i-1}: (B, H)
            scores = self.v(torch.tanh(keys + self.U(s_prev).unsqueeze(1))).squeeze(-1)
            scores = scores.masked_fill(~source_mask, -torch.inf)
            weights = torch.softmax(scores, dim=-1)               # (B, S), sums to 1
            context = torch.bmm(weights.unsqueeze(1), encoder_outputs)          # (B, 1, H)
            merged = torch.cat([embedded[:, t:t+1], context], dim=-1)            # (B, 1, E+H)
            out, (hidden, cell) = self.lstm(merged, (hidden, cell))
            outputs.append(out)
            all_weights.append(weights)
        outputs = torch.cat(outputs, dim=1)
        return self.output(outputs), (hidden, cell), torch.stack(all_weights, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size,
                 embedding_dim=128, hidden_dim=256, attention=False, bidirectional=False):
        super().__init__()
        self.attention = bool(attention)
        self.bidirectional = bool(bidirectional)
        self.encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim, self.bidirectional)
        decoder_class = AttentionDecoder if self.attention else Decoder
        self.decoder = decoder_class(target_vocab_size, embedding_dim, hidden_dim)

    def _encode(self, source_ids, source_lengths):
        """Return (decoder initial state, memory). memory is None for baseline."""
        if not self.attention:
            return self.encoder(source_ids, source_lengths), None
        outputs, state = self.encoder(source_ids, source_lengths, return_outputs=True)
        positions = torch.arange(source_ids.size(1), device=source_ids.device)
        mask = positions.unsqueeze(0) < source_lengths.to(source_ids.device).unsqueeze(1)
        return state, (outputs, mask)

    def _decode(self, token_ids, state, memory):
        if memory is None:
            logits, state = self.decoder(token_ids, state)
            return logits, state, None
        return self.decoder(token_ids, state, *memory)

    def forward(self, source_ids, source_lengths, decoder_input_ids, return_attention=False):
        """100% teacher forcing; output shape: batch, target steps, vocabulary."""
        state, memory = self._encode(source_ids, source_lengths)
        logits, _, weights = self._decode(decoder_input_ids, state, memory)
        return (logits, weights) if return_attention else logits

    @torch.no_grad()
    def generate(self, source_ids, source_lengths, max_new_tokens=64,
                 return_attention=False):
        """Greedy decoding; return token lists excluding SOS/EOS plus EOS flags.

        PAD/SOS/UNK are suppressed during generation. Outputs are not executed.
        A missing EOS flag means generation reached the length limit.
        With return_attention=True (attention model only), a third value holds,
        per example, one source-attention row per generated token.
        """
        if return_attention and not self.attention:
            raise ValueError('return_attention requires attention=True')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        was_training = self.training
        self.eval()
        try:
            batch_size = source_ids.size(0)
            state, memory = self._encode(source_ids, source_lengths)
            attention_rows = [[] for _ in range(batch_size)]
            previous = torch.full((batch_size, 1), SOS_ID,
                                  dtype=torch.long, device=source_ids.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)
            result = [[] for _ in range(batch_size)]
            for _ in range(max_new_tokens):
                logits, state, weights = self._decode(previous, state, memory)
                scores = logits[:, -1, :].clone()
                scores[:, [PAD_ID, UNK_ID, SOS_ID]] = -torch.inf
                next_ids = scores.argmax(dim=-1)
                next_ids = torch.where(finished, EOS_ID, next_ids)
                for i, next_id in enumerate(next_ids.tolist()):
                    if not finished[i].item() and next_id != EOS_ID:
                        result[i].append(next_id)
                        if return_attention:
                            length = int(source_lengths[i])
                            attention_rows[i].append(weights[i, -1, :length].tolist())
                finished |= next_ids.eq(EOS_ID)
                if finished.all().item():
                    break
                previous = next_ids.unsqueeze(1)
            if return_attention:
                return result, finished.tolist(), attention_rows
            return result, finished.tolist()
        finally:
            self.train(was_training)


def smoke_test():
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent / 'processed'
    if not (root / 'vocabularies.json').exists():
        raise SystemExit('Run preprocess.py first, in this project folder.')
    vocab = json.loads((root / 'vocabularies.json').read_text(encoding='utf-8'))
    for name in ['source_token_to_id', 'target_token_to_id']:
        assert [vocab[name][t] for t in ['<PAD>','<UNK>','<SOS>','<EOS>']] == [0,1,2,3]
    rows = []
    with (root / 'train.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == 8:
                break
    if not rows:
        raise SystemExit('Training dataset is empty.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = collate_batch(rows)
    source = batch['source_ids'].to(device)
    lengths = batch['source_lengths']
    inputs = batch['decoder_input_ids'].to(device)
    labels = batch['decoder_target_ids'].to(device)
    for bidirectional in (False, True):
        for attention in (False, True):
            torch.manual_seed(42)
            print(f"\n=== Variant: {'bi' if bidirectional else 'uni'}-encoder, "
                  f"{'attention' if attention else 'no attention'} ===")
            check_variant(attention, vocab, rows, device, source, lengths, inputs, labels,
                          bidirectional)


def check_variant(attention, vocab, rows, device, source, lengths, inputs, labels,
                  bidirectional=False):
    model = Seq2Seq(len(vocab['source_token_to_id']),
                    len(vocab['target_token_to_id']), attention=attention,
                    bidirectional=bidirectional).to(device)
    logits = model(source, lengths, inputs)
    assert logits.shape == (*labels.shape, len(vocab['target_token_to_id']))
    loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
               for p in model.parameters())
    assert model.encoder.embedding.weight.grad[PAD_ID].abs().sum().item() == 0
    # Appending padding must not change the encoder's final states.
    with torch.no_grad():
        extended = nn.functional.pad(source, (0, 3), value=PAD_ID)
        original_state = model.encoder(source, lengths)
        padded_state = model.encoder(extended, lengths)
        for original, padded in zip(original_state, padded_state):
            torch.testing.assert_close(original, padded)
    generated, flags = model.generate(source, lengths, max_new_tokens=5)
    assert len(generated) == len(rows)
    assert all(len(x) <= 5 for x in generated)
    assert model.training  # generate restores the caller's mode
    # Force EOS to verify termination independently of random model outputs.
    with torch.no_grad():
        saved_weight = model.decoder.output.weight.clone()
        saved_bias = model.decoder.output.bias.clone()
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.fill_(-100)
        model.decoder.output.bias[EOS_ID] = 100
        stopped, eos_flags = model.generate(source, lengths, max_new_tokens=5)
        assert all(eos_flags) and all(not x for x in stopped)
        model.decoder.output.weight.copy_(saved_weight)
        model.decoder.output.bias.copy_(saved_bias)
    print('Device:', device)
    print('Source batch shape:', tuple(source.shape))
    print('Decoder input shape:', tuple(inputs.shape))
    print('Logits shape:', tuple(logits.shape))
    print('Trainable parameters:', sum(p.numel() for p in model.parameters()))
    print(f'Initial loss: {loss.item():.4f}')
    print('Forward pass: OK')
    print('Backward pass and finite gradients: OK')
    print('Padding invariance: OK')
    if attention:
        with torch.no_grad():
            model.eval()
            logits_a, weights = model(source, lengths, inputs, return_attention=True)
            logits_b, weights_b = model(extended, lengths, inputs, return_attention=True)
            torch.testing.assert_close(logits_a, logits_b)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
            pad_positions = torch.arange(extended.size(1), device=device)[None, :] >= lengths.to(device)[:, None]
            assert weights_b.masked_select(pad_positions[:, None, :].expand_as(weights_b)).abs().max().item() == 0
            # Step-by-step greedy must agree with one teacher-forced pass fed
            # with its own outputs (same recurrence, same weights).
            out, _, rows_att = model.generate(source, lengths, max_new_tokens=6, return_attention=True)
            for i, (tokens, att) in enumerate(zip(out, rows_att)):
                assert len(att) == len(tokens)
                if not tokens:
                    continue
                prefix = torch.tensor([[SOS_ID] + tokens[:-1]], device=device)
                step_logits, step_weights = model(source[i:i+1], lengths[i:i+1], prefix,
                                                  return_attention=True)
                step_logits[..., [PAD_ID, UNK_ID, SOS_ID]] = -torch.inf
                assert step_logits.argmax(-1)[0].tolist() == tokens
                torch.testing.assert_close(step_weights[0, :, :int(lengths[i])],
                                           torch.tensor(att, device=device))
            model.train()
        print('Attention: rows sum to 1, PAD masked, padding-invariant logits: OK')
        print('Attention: greedy step-by-step == teacher-forced pass: OK')
    print('Greedy decoding and EOS stopping: OK')
    print('Variant passed. Model has NOT been trained or saved.')


if __name__ == '__main__':
    smoke_test()
