# Python code generation with LSTM encoder–decoders

NLP course project: generating a short Python function from an English instruction
with encoder–decoder models implemented and trained from scratch in PyTorch. No
pretrained model or model API is used.

```
"Write solve(numbers) that will count the values in numbers that are at most 4."
        → def solve(numbers):
              return sum(1 for n in numbers if n <= 4)
```

The analysis scripts below write results, figures and examples to `results_attention_comparison/`.

## Setup

Python 3.10+. Dataset generation and preprocessing need no third-party packages.
Training/evaluation need PyTorch (2.7+ tested; CPU or CUDA). Figures need matplotlib.

```bash
pip install -r requirements.txt
```

## Interactive demo (Streamlit)

```bash
pip install -r requirements.txt     # torch (CPU), streamlit, matplotlib
streamlit run app.py                # opens http://localhost:8501
```

`runs/` contains one trained model per architecture (seed 42), so the app works straight away.

`app.py` is a guided demo. Build an instruction step by step (only phrasings the model was trained
on) or type your own: unknown words and numbers outside −5…10 are flagged as you type, and six
example buttons show successes and known limits. **Generate** jumps to the result: the generated
function, what the model understood (operation, comparison, number), its confidence, and checks
against the program the sentence describes (bounded interpreter, never `exec`). Below, collapsible
sections show each stage: tokens and ids, embeddings, encoder states, every decoding step with its
top candidates and attention, the attention map, and the same sentence through all four
architectures. Pick the architecture and training run in the sidebar.

## Pipeline and files

| Stage | File | What it does |
|---|---|---|
| Dataset | `generate_dataset.py` | Builds `data/*.jsonl` from templates and checks every reference program against behavioural tests |
| Preprocessing | `preprocess.py` | Tokenizes, builds train-only vocabularies, encodes ids, writes `processed/` |
| Model | `model.py` | Encoder (uni/bidirectional LSTM), decoder (plain or Bahdanau attention), greedy decoding; `python model.py` runs the structural tests |
| Training | `train.py` | Teacher-forced training, validation-loss checkpoint selection, early stopping |
| Evaluation | `evaluate.py` | Greedy generation on both test splits; syntax, behavioural tests, condition correctness, token accuracy, perplexity |
| Inference | `generate.py`, `app.py` | Command-line generation (`--trace` prints each step); Streamlit app visualising every stage |
| Analysis | `compare.py`, `attention_alignment.py`, `make_figures.py`, `select_examples.py` | Comparison tables, attention diagnostics, figures, qualitative examples |

## Reproduce

```bash
python generate_dataset.py          # data/*.jsonl, data/summary.json
python preprocess.py                # processed/*.jsonl, vocabularies.json, report.json
python model.py                     # structural tests for all 4 variants

python train.py --overfit 16 --epochs 400 [--bidirectional] [--attention]   # sanity check
python train.py --seed 42                                   # baseline
python train.py --seed 42 --attention                       # + attention
python train.py --seed 42 --bidirectional                   # + bidirectional encoder
python train.py --seed 42 --bidirectional --attention       # both
# (the reported experiment repeats each with --seed 7, 123, 1, 2)

python evaluate.py --checkpoint runs/<RUN>/<best_checkpoint from runs/<RUN>/summary.json>
python compare.py runs/*_seed* --out results_attention_comparison
python attention_alignment.py runs/attention_seed* runs/bi_attention_seed* --out results_attention_comparison/attention_alignment.json
python make_figures.py runs/*_seed* --out results_attention_comparison/figures
python select_examples.py --out results_attention_comparison/examples.md

python generate.py --checkpoint runs/<RUN>/<checkpoint>.pt --trace "Write solve(numbers) that will count the values in numbers that are at most 4."
```

Every `train.py` call creates a new `runs/<variant>_seed<seed>_<timestamp>/` folder with
`config.json` (all settings, file hashes), `history.csv` (losses per epoch), the improved
checkpoints and `summary.json`. Each `evaluate.py` call adds an `evaluation_<timestamp>/` folder;
the analysis scripts use the most recent one. Seeds reduce variability, but results are not
identical across devices (CPU vs CUDA).

## Dataset

Synthetic and template-based, created for this project (no MBPP, CoNaLa or CodeSearchNet data).
Each example: `instruction` (model input), `code` (target), `tests` (behavioural checks; never
fed to the model), plus metadata (`task_id`, `family`, `structure`, `wording_id`).

| Split | Examples | Distinct tasks | Purpose |
|---|---:|---:|---|
| train | 2000 | 250 | 4 instruction wrappers × 2 goal phrasings |
| validation_wording | 500 | 250 | Same tasks, unseen wrapper; checkpoint selection |
| test_wording | 500 | 250 | Same tasks, another unseen wrapper |
| test_composition | 96 | 48 | Unseen operation + relation combinations, familiar wrapper |

- 298 distinct tasks, 28 program structures; list tasks dominate (1,920 of 2,000 training examples).
- `test_composition` withholds sum + `>`, count + `<=` and filter + `!=`; every operation and
  relation appears elsewhere in training. This is the split that tests generating programs not
  seen during training.
- Wording splits share programs with training: they test robustness to new wording, not new programs.
- Thresholds are −5…10 in every split (no extrapolation to unseen constants).
- Do not merge and re-split the files; do not tune on either test split.

## Preprocessing (summary)

- Instructions: lowercased, split into words, digits and punctuation (`-3` → `-`, `3`).
  97-token source vocabulary, built on training only; unseen words → `<UNK>`
  (≈9% of tokens in the wording splits: the wrapper words).
- Code: Python's own `tokenize` module; tokens keep their type (`NAME:sum`, `OP:<=`), with
  `<NEWLINE>`, `<INDENT>`, `<DEDENT>` so indentation can be rebuilt. 51-token target vocabulary.
- Special tokens `<PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3`. Source gets `<EOS>`; decoder input is
  `<SOS> + code`, decoder target is `code + <EOS>`.
- Padding is per batch (right-padded, PAD ignored by the loss, packed sequences in the encoder,
  masked in attention). No truncation: longest source is 33 tokens and longest target 25, so
  every function stays complete.

## Model and training (summary)

Embeddings 128-d; one LSTM layer, hidden 256 (bidirectional: 128 per direction, concatenated);
decoder initial state = encoder final state. Attention: Bahdanau (additive), as in the course:
e_ij = v^T tanh(W h_j + U s_(i-1)), softmax over the real source positions, context vector
c_i = sum_j a_ij h_j concatenated with the input embedding before the decoder LSTM.
Parameters: 822,579 (baseline), 1,216,307 (attention), 691,507 (bidirectional),
1,085,235 (bidirectional + attention).
Adam (lr 1e-3), batch 32, ≤30 epochs, early stopping patience 5 on validation loss, token
cross-entropy (PAD ignored), gradient clipping 1.0, teacher forcing ratio 1.0. Greedy decoding,
max 64 tokens, PAD/SOS/UNK never generated.

## Safety of evaluation

Generated code is never passed to `exec`/`eval`. `evaluate.py` checks behaviour with a bounded
AST interpreter that supports only this dataset's expression subset; anything else is counted
as unsupported. `generate_dataset.py` executes only its own trusted reference templates.

## Known limitations

- Synthetic data from one generator; 3 held-out combinations of 32 examples each.
- Without attention, unseen wrapper words can turn a count into a sum; with attention the only
  remaining errors are in the tiny arithmetic family.
- String, arithmetic and condition families are tiny (32/32/16 training examples).
- `generate_dataset.py` writes platform line endings, so file hashes differ between Windows
  and Linux even though the records are identical.
