"""Instruction → Python: an interactive demo of the encoder–decoder.

    pip install -r requirements.txt
    streamlit run app.py

Build or type an instruction, press Generate, and the page jumps to the generated
function, what the model understood and an automatic check. Every internal stage
(tokens, embeddings, encoder, each decoding step, attention, all architectures)
is one click away below the result. Generated code is NEVER executed with
exec/eval: checks use the bounded AST interpreter from evaluate.py.
"""
import ast
import html
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

from evaluate import RestrictedPython, Unsupported, parse_condition, same_value
from model import Seq2Seq, PAD_ID, UNK_ID, SOS_ID, EOS_ID
from preprocess import tokenize_instruction, encode, detokenize_code

ROOT = Path(__file__).resolve().parent
INK, MUTED, ACCENT, WARM, GOOD, BAD = '#152238', '#5B6577', '#2A78D6', '#EB6834', '#1E7F4F', '#B42318'

VARIANT_LABEL = {'bi_attention': 'Bidirectional encoder + Bahdanau attention',
                 'baseline': 'Baseline LSTM (no attention)',
                 'attention': 'LSTM + Bahdanau attention',
                 'bi_baseline': 'Bidirectional encoder, no attention'}
VARIANT_ORDER = ['attention', 'baseline', 'bi_baseline', 'bi_attention']

# ---- what the model was trained on (mirrors generate_dataset.py) ----------------------------
WRAPPERS = {
    'Write … that will …': ('Write {sig} that will {goal}.', True),
    'Implement … to …': ('Implement {sig} to {goal}.', True),
    'Create …; it should …': ('Create {sig}; it should {goal}.', True),
    'Define … to …': ('Define {sig} to {goal}.', True),
    'The function … must … (never seen in training)': ('The function {sig} must {goal}.', False),
    'Using the signature …, … (never seen in training)': ('Using the signature {sig}, {goal}.', False),
}
COMPARISONS = {  # phrase -> (symbol, relation id)
    'greater than': ('>', 'gt'), 'above': ('>', 'gt'),
    'greater than or equal to': ('>=', 'ge'), 'at least': ('>=', 'ge'),
    'less than': ('<', 'lt'), 'below': ('<', 'lt'),
    'less than or equal to': ('<=', 'le'), 'at most': ('<=', 'le'),
    'equal to': ('==', 'eq'), 'exactly': ('==', 'eq'),
    'not equal to': ('!=', 'ne'), 'different from': ('!=', 'ne'),
}
LIST_TASKS = {
    'Keep the matching values (filter)': ('filter', 'return, in their original order and retaining duplicates, the values in numbers that are {cmp} {k}',
                                          '[n for n in numbers if n {sym} {k}]'),
    'Count the matching values': ('count', 'count the values in numbers that are {cmp} {k}',
                                  'sum(1 for n in numbers if n {sym} {k})'),
    'Sum the matching values': ('sum', 'return the sum of the values in numbers that are {cmp} {k}',
                                'sum(n for n in numbers if n {sym} {k})'),
}
HELD_OUT = {('sum', 'gt'), ('count', 'le'), ('filter', 'ne')}
OTHER_TASKS = {
    'Numbers': {
        'Add two numbers': ('solve(a, b)', ['return the sum of a and b', 'add a and b'], 'a + b'),
        'Subtract': ('solve(a, b)', ['subtract b from a', 'return a minus b'], 'a - b'),
        'Multiply': ('solve(a, b)', ['return the product of a and b', 'multiply a by b'], 'a * b'),
        'Square': ('solve(n)', ['return the square of n', 'multiply n by itself'], 'n * n'),
        'Is it even?': ('solve(n)', ['check whether integer n is even', 'return whether n is divisible by 2'], 'n % 2 == 0'),
        'Is it odd?': ('solve(n)', ['check whether integer n is odd', 'return whether n is not divisible by 2'], 'n % 2 != 0'),
    },
    'Text': {
        'Reverse': ('solve(text)', ['reverse the string text', 'return text in reverse order'], 'text[::-1]'),
        'Lowercase': ('solve(text)', ['convert text to lowercase', 'return a lowercase version of text'], 'text.lower()'),
        'Uppercase': ('solve(text)', ['convert text to uppercase', 'return an uppercase version of text'], 'text.upper()'),
        'Count a character': ('solve(text, char)', ['count occurrences of the single character char in text, respecting case',
                                                    'return how many times the single character char appears in text, case-sensitively'],
                              'text.count(char)'),
    },
}
SAMPLE_ARGS = {'numbers': lambda k: [[[]], [[k]], [[k - 1, k, k + 1]], [[k + 1, k + 1, k - 1]], [[-9, 0, 13]], [[3, 5, 5, -2, 9]]],
               'a,b': lambda k: [[3, 4], [-2, 2], [0, 0], [7, -3]],
               'n': lambda k: [[-3], [0], [1], [2], [9]],
               'text': lambda k: [[''], ['a'], ['Hello World'], ['AbBa']],
               'text,char': lambda k: [['Banana', 'a'], ['', 'x'], ['AaA', 'a'], ['abc', 'z']]}
EXAMPLES = [
    ('Held-out pair · filter + not equal',
     'Create solve(numbers); it should return, in their original order and retaining duplicates, the values in numbers that are different from 3.'),
    ('Held-out pair · count + at most', 'Define solve(numbers) to count the values in numbers that are less than or equal to 7.'),
    ('Unknown word “please”', 'Please write solve(numbers) that will count the values in numbers that are at most 4.'),
    ('New threshold −9', 'Write solve(numbers) that will return the sum of the values in numbers that are above -9.'),
    ('Limit · number 25 is unknown', 'Write solve(numbers) that will count the values in numbers that are at most 25.'),
    ('Limit · “bigger” is unknown', 'Write solve(numbers) that will count the values in numbers that are bigger than 2.'),
]

st.set_page_config(page_title='Instruction → Python', page_icon='🧩', layout='wide')
st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; max-width: 1180px;}}
.hero {{background: {INK}; color: #F6F5F1; border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;}}
.hero h1 {{color: #F6F5F1; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0;}}
.hero p {{color: #C9D3E3; font-size: 1.05rem; margin: 0;}}
.hero .flow {{margin-top: 14px; font-family: ui-monospace, Menlo, monospace; font-size: .9rem; color: #8FB4E8;}}
.chip {{display: inline-block; padding: 3px 11px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: .92rem;
        background: #EEF2F7; color: {INK}; border: 1px solid #DCE3EE;}}
.chip.good {{background: #E7F4EC; border-color: #BFE3CD; color: {GOOD};}}
.chip.bad {{background: #FCEBEA; border-color: #F4C7C3; color: {BAD};}}
.chip.warn {{background: #FFF4E5; border-color: #F7D9A8; color: #8A4B08;}}
.chip.unk {{background: #FCEBEA; border-color: #F4C7C3; color: {BAD}; font-weight: 600;}}
.preview {{background: #FFFFFF; border: 1px dashed #B9C6DA; border-radius: 12px; padding: 12px 16px; font-size: 1.05rem;}}
.section-label {{font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; font-weight: 600;
                 margin: 6px 0 2px 0;}}
div[data-testid="stCode"] pre {{font-size: 1.05rem;}}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_data
def find_runs():
    """Trained runs with a usable checkpoint (overfit diagnostics excluded)."""
    runs = []
    for config_path in sorted((ROOT / 'runs').glob('*/config.json')):
        run = config_path.parent
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('mode') == 'overfit':
            continue
        checkpoint = None
        summary_path = run / 'summary.json'
        if summary_path.exists():
            name = json.loads(summary_path.read_text(encoding='utf-8')).get('best_checkpoint')
            if name and (run / name).exists():
                checkpoint = run / name
        if checkpoint is None:
            candidates = sorted(run.glob('checkpoint_epoch_*.pt'))
            checkpoint = candidates[-1] if candidates else None
        if checkpoint is None:
            continue
        model_config = config.get('model_config', {})
        variant = config.get('variant') or (('bi_' if model_config.get('bidirectional') else '')
                                            + ('attention' if model_config.get('attention') else 'baseline'))
        runs.append({'run': run.name, 'variant': variant, 'seed': config.get('seed'),
                     'device': config.get('device', '?'), 'checkpoint': str(checkpoint)})
    return runs


@st.cache_resource
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = Seq2Seq(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['vocabularies'], checkpoint.get('epoch')


@st.cache_data
def load_references():
    references = {}
    for split in ['test_composition', 'test_wording', 'validation_wording', 'train']:
        path = ROOT / 'data' / f'{split}.jsonl'
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    row = json.loads(line)
                    references.setdefault(row['instruction'].strip(), {**row, 'split': split})
    return references


# ---------------------------------------------------------------- model, step by step
@torch.no_grad()
def run_pipeline(model, vocab, instruction, max_new_tokens=64, top_k=5):
    words = tokenize_instruction(instruction)
    source_tokens = words + ['<EOS>']
    source_ids = encode(words, vocab['source_token_to_id']) + [EOS_ID]
    source, lengths = torch.tensor([source_ids]), torch.tensor([len(source_ids)])
    embedded = model.encoder.embedding(source)[0]
    encoder_outputs, (hidden, _) = model.encoder(source, lengths, return_outputs=True)
    state, memory = model._encode(source, lengths)
    id_to_token = vocab['target_id_to_token']
    previous, steps, output = torch.tensor([[SOS_ID]]), [], []
    for step in range(max_new_tokens):
        logits, state, weights = model._decode(previous, state, memory)
        scores = logits[0, -1].clone()
        scores[[PAD_ID, UNK_ID, SOS_ID]] = -torch.inf
        probabilities = torch.softmax(scores, dim=-1)
        top = torch.topk(probabilities, top_k)
        chosen = int(probabilities.argmax())
        steps.append({'step': step, 'input': id_to_token[int(previous[0, 0])], 'token': id_to_token[chosen],
                      'probability': float(probabilities[chosen]),
                      'top': [(id_to_token[int(i)], float(p)) for p, i in zip(top.values, top.indices)],
                      'attention': weights[0, -1].tolist() if weights is not None else None})
        if chosen == EOS_ID:
            break
        output.append(id_to_token[chosen])
        previous = torch.tensor([[chosen]])
    try:
        code, rebuild_error = detokenize_code(output), None
    except (ValueError, IndexError, IndentationError, SyntaxError) as error:
        code, rebuild_error = None, str(error)
    return {'words': words, 'source_tokens': source_tokens, 'source_ids': source_ids,
            'unknown': [w for w in words if w not in vocab['source_token_to_id']],
            'embedded': embedded.tolist(), 'encoder_outputs': encoder_outputs[0].tolist(),
            'final_hidden': hidden[-1, 0].tolist(),
            'steps': steps, 'output_tokens': output,
            'ended': bool(steps) and steps[-1]['token'] == '<EOS>', 'code': code, 'rebuild_error': rebuild_error}


def infer_expected(instruction):
    """Rule-based reading of a sentence written in the dataset's phrasing.

    Returns (program, threshold) for the function the sentence describes, or (None, 0)
    when the sentence is not phrased the way the dataset phrases tasks. Independent
    of the neural model: it is only used to check the model's answer.
    """
    text = ' '.join(instruction.lower().split())
    if 'solve(numbers)' in text:
        operation = ('count' if 'count the values' in text else
                     'sum' if 'sum of the values' in text else
                     'filter' if 'the values in numbers' in text else None)
        for phrase in sorted(COMPARISONS, key=len, reverse=True):
            match = re.search(rf'that are {re.escape(phrase)} (-?\d+)\b', text)
            if operation and match:
                symbol, k = COMPARISONS[phrase][0], int(match.group(1))
                inner = {'filter': f'[n for n in numbers if n {symbol} {k}]',
                         'count': f'sum(1 for n in numbers if n {symbol} {k})',
                         'sum': f'sum(n for n in numbers if n {symbol} {k})'}[operation]
                return f'def solve(numbers):\n    return {inner}\n', k
        return None, 0
    for tasks in OTHER_TASKS.values():
        for signature, goals, body in tasks.values():
            if signature in text and any(goal in text for goal in goals):
                return f'def {signature}:\n    return {body}\n', 0
    return None, 0


def syntax_error_of(code):
    try:
        compile(code, '<generated>', 'exec')   # compiled only, never run
        return None
    except (SyntaxError, ValueError, TypeError) as error:
        return str(error)


def run_safely(code, args):
    return RestrictedPython(code).run(args)


def argument_names(code):
    try:
        return [a.arg for a in ast.parse(code).body[0].args.args]
    except Exception:
        return []


def check_against(code, expected_code, cases):
    rows = []
    for args in cases:
        try:
            expected = run_safely(expected_code, args)
        except Exception:
            continue
        try:
            got = run_safely(code, args)
            ok = same_value(got, expected)
            got_text = repr(got)
        except Unsupported as error:
            ok, got_text = False, f'unsupported: {error}'
        except Exception as error:
            ok, got_text = False, f'{type(error).__name__}: {error}'
        rows.append({'input': json.dumps(args), 'expected': repr(expected), 'got': got_text, 'ok': ok})
    return rows


# ---------------------------------------------------------------- charts
def heatmap(matrix, row_labels, row_title, column_title, title):
    frame = pd.DataFrame([{'row': f'{r:02d} {label}', 'col': c, 'value': v}
                          for r, (label, values) in enumerate(zip(row_labels, matrix)) for c, v in enumerate(values)])
    limit = max(abs(frame['value'].min()), abs(frame['value'].max()), 1e-6)
    return alt.Chart(frame, title=title).mark_rect().encode(
        x=alt.X('col:O', title=column_title, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('row:N', title=row_title, sort=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='blueorange', domain=[-limit, limit]), title='value'),
        tooltip=[alt.Tooltip('row:N', title=row_title), alt.Tooltip('col:O', title=column_title),
                 alt.Tooltip('value:Q', format='.3f')],
    ).properties(height=max(90, 26 * len(row_labels)))


def bars(frame, x, y, title, highlight=None):
    color = (alt.condition(alt.datum[y] == highlight, alt.value(WARM), alt.value(ACCENT))
             if highlight is not None else alt.value(ACCENT))
    base = alt.Chart(frame, title=title).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f'{x}:Q', scale=alt.Scale(domain=[0, 1]), title=None), y=alt.Y(f'{y}:N', sort=None, title=None),
        color=color, tooltip=[y, alt.Tooltip(f'{x}:Q', format='.3f')])
    labels = alt.Chart(frame).mark_text(align='left', dx=4, color=INK).encode(
        x=f'{x}:Q', y=alt.Y(f'{y}:N', sort=None), text=alt.Text(f'{x}:Q', format='.3f'))
    return (base + labels).properties(height=alt.Step(28))


def attention_map(steps, source_tokens):
    frame = pd.DataFrame([{'generated': f"{s['step']:02d} {s['token']}", 'source': f'{j:02d} {source_tokens[j]}',
                           'weight': w} for s in steps for j, w in enumerate(s['attention'])])
    return alt.Chart(frame).mark_rect().encode(
        x=alt.X('source:N', sort=None, title='Your sentence (tokens)', axis=alt.Axis(labelAngle=-50, labelOverlap=False)),
        y=alt.Y('generated:N', sort=None, title='Generated code (in order)', axis=alt.Axis(labelOverlap=False)),
        color=alt.Color('weight:Q', scale=alt.Scale(scheme='blues', domain=[0, 1]), title='attention'),
        tooltip=['generated', 'source', alt.Tooltip('weight:Q', format='.3f')],
    ).properties(height=max(260, 26 * len(steps)))


def chips(items):
    return ' '.join(f"<span class='chip {kind}'>{html.escape(str(text))}</span>" for text, kind in items)


# ---------------------------------------------------------------- sidebar
runs = find_runs()
if not runs:
    st.error('No trained model found in runs/. Train one first: `python train.py --attention`.')
    st.stop()
references = load_references()

with st.sidebar:
    st.markdown('### Model')
    variants = [v for v in VARIANT_ORDER if any(r['variant'] == v for r in runs)]
    variant = st.selectbox('Architecture', variants,
                           format_func=lambda v: VARIANT_LABEL.get(v, v) + (' · recommended' if v == 'attention' else ''))
    options = sorted([r for r in runs if r['variant'] == variant],
                     key=lambda r: (r['seed'] != 42, r['seed'], r['device'] != 'cpu'))
    chosen_run = st.selectbox('Training run', options,
                              format_func=lambda r: f"seed {r['seed']}" + (' · original GPU run' if r['device'].startswith('cuda') else ''))
    seed = chosen_run['seed']
    model, vocab, epoch = load_model(chosen_run['checkpoint'])
    st.caption(f"{sum(p.numel() for p in model.parameters()):,} parameters · checkpoint epoch {epoch}")
    with st.expander('Decoding settings'):
        max_new_tokens = st.slider('Max generated tokens', 8, 64, 64)
        top_k = st.slider('Alternatives shown per step', 3, 10, 5)
    st.markdown('### About')
    st.caption('A small encoder–decoder trained from scratch (PyTorch) on a synthetic dataset of short Python '
               'functions. It knows list filters / counts / sums with a comparison and a number from −5 to 10, '
               'basic arithmetic, odd/even, and four text tasks — nothing else. Generated code is never executed; '
               'checks run in a restricted interpreter.')

# ---------------------------------------------------------------- hero + input
st.markdown(f"""<div class='hero'><h1>Instruction → Python</h1>
<p>Describe a small function in English. A neural encoder–decoder trained from scratch writes the code.</p>
<div class='flow'>your sentence → tokens → embeddings → encoder → decoder, one token at a time → Python</div></div>""",
            unsafe_allow_html=True)

if 'free_text' not in st.session_state:
    st.session_state.free_text = EXAMPLES[0][1]


def use_example(text):
    st.session_state.free_text = text
    st.session_state.mode = 'Write my own'
    expected, k = infer_expected(text)
    st.session_state.request = {'instruction': text, 'expected': expected, 'note': None, 'k': k}
    st.session_state.scroll = True


mode = st.radio('How do you want to write the instruction?', ['Build it step by step', 'Write my own'],
                horizontal=True, key='mode', label_visibility='collapsed')

if mode == 'Build it step by step':
    st.markdown("<div class='section-label'>1 · What should the function do?</div>", unsafe_allow_html=True)
    family = st.segmented_control('Kind of task', ['Lists of numbers', 'Numbers', 'Text'], default='Lists of numbers',
                                  label_visibility='collapsed') or 'Lists of numbers'
    expected_code, note, k = None, None, 0
    if family == 'Lists of numbers':
        c1, c2, c3 = st.columns([1.3, 1.2, 1])
        task = c1.selectbox('Operation', list(LIST_TASKS))
        phrase = c2.selectbox('Comparison', list(COMPARISONS))
        k = c3.number_input('Number (−5 to 10)', -5, 10, 3, help='The model only saw thresholds from −5 to 10.')
        operation, goal_template, code_template = LIST_TASKS[task]
        symbol, relation = COMPARISONS[phrase]
        signature, goal = 'solve(numbers)', goal_template.format(cmp=phrase, k=k)
        expected_code = f"def solve(numbers):\n    return {code_template.format(sym=symbol, k=k)}\n"
        if (operation, relation) in HELD_OUT:
            note = (f'**{operation} + `{symbol}` never appears together in training**: '
                    'this checks whether the model can combine pieces it learned separately.')
    else:
        c1, c2 = st.columns([1.3, 2.2])
        task = c1.selectbox('Task', list(OTHER_TASKS[family]))
        signature, goals, body = OTHER_TASKS[family][task]
        goal = c2.selectbox('Phrasing', goals)
        expected_code = f'def {signature}:\n    return {body}\n'
    st.markdown("<div class='section-label'>2 · How should the sentence start?</div>", unsafe_allow_html=True)
    wrapper_name = st.selectbox('Sentence pattern', list(WRAPPERS), label_visibility='collapsed')
    pattern, familiar = WRAPPERS[wrapper_name]
    instruction = pattern.format(sig=signature, goal=goal)
    if not familiar:
        st.caption('This sentence pattern was held out of training: its first words are unknown to the model.')
    st.markdown("<div class='section-label'>Your instruction</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='preview'>{html.escape(instruction)}</div>", unsafe_allow_html=True)
    if note:
        st.info(note)
else:
    st.markdown("<div class='section-label'>Try an example</div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES):
        columns[i % 3].button(label, key=f'ex{i}', on_click=use_example, args=(text,), width='stretch')
    instruction = st.text_area('Your instruction', key='free_text', height=90)
    expected_code, note = None, None
    # live guidance: what the model will not understand
    words = tokenize_instruction(instruction)
    unknown = sorted({w for w in words if w not in vocab['source_token_to_id']})
    numbers = [int(m) for m in re.findall(r'-?\d+', instruction)]
    issues = []
    if 'solve' not in words:
        issues.append(('Start with a signature such as solve(numbers), solve(n), solve(a, b) or solve(text)', 'warn'))
    for w in unknown:
        issues.append((f'unknown word: {w}', 'unk'))
    for n in numbers:
        if not -5 <= n <= 10:
            issues.append((f'{n} is outside −5…10', 'warn'))
    if issues:
        st.markdown(chips(issues), unsafe_allow_html=True)
        st.caption('Unknown words become `<UNK>`: the model sees that a word was there, not what it meant.')
    elif instruction.strip():
        st.markdown(chips([('every word is in the model’s vocabulary', 'good')]), unsafe_allow_html=True)
    with st.expander('How to write an instruction the model understands'):
        st.markdown("""
- **Name the function and its input first:** `solve(numbers)`, `solve(n)`, `solve(a, b)`, `solve(text)` or `solve(text, char)`.
- **Use a pattern it knows:** “Write solve(numbers) that will …”, “Implement solve(numbers) to …”, “Create solve(numbers); it should …”, “Define solve(numbers) to …”.
- **For lists, say the operation:** “return, in their original order and retaining duplicates, the values in numbers that are …”, “count the values in numbers that are …”, “return the sum of the values in numbers that are …”.
- **Then a comparison and a number from −5 to 10:** greater than / above, greater than or equal to / at least, less than / below, less than or equal to / at most, equal to / exactly, not equal to / different from.
- **Other tasks:** “return the sum of a and b”, “subtract b from a”, “multiply a by b”, “return the square of n”, “check whether integer n is even / odd”, “reverse the string text”, “convert text to lowercase / uppercase”.

Anything else is outside what this small model has learned, which is exactly what the limit examples show.
""")

if st.button('Generate code  →', type='primary', disabled=not instruction.strip()):
    if expected_code is None:
        expected_code, k = infer_expected(instruction)
    st.session_state.request = {'instruction': instruction.strip(), 'expected': expected_code, 'note': note, 'k': k}
    st.session_state.scroll = True

request = st.session_state.get('request')
if not request:
    st.stop()

# ---------------------------------------------------------------- result
result = run_pipeline(model, vocab, request['instruction'], max_new_tokens, top_k)
reference = references.get(request['instruction'])
expected_code = request.get('expected') or (reference['code'] if reference else None)

st.markdown("<div id='result'></div>", unsafe_allow_html=True)
st.divider()
st.markdown('## Result')
st.caption(f"“{request['instruction']}” · {VARIANT_LABEL.get(variant, variant)}, seed {seed}")

left, right = st.columns([1.35, 1])
with left:
    if result['code'] is None:
        st.error('The model produced tokens that do not form a function.')
        st.code(' '.join(t.split(':')[-1] for t in result['output_tokens']), language=None)
        syntax_error = 'cannot be rebuilt'
    else:
        st.code(result['code'], language='python')
        syntax_error = syntax_error_of(result['code'])
    if not result['ended']:
        st.warning('Generation stopped at the length limit without an end token.')

with right:
    status = []
    status.append(('valid Python', 'good') if not syntax_error else ('invalid Python', 'bad'))
    parsed = parse_condition(result['output_tokens'])
    if parsed:
        st.markdown("<div class='section-label'>What the model understood</div>", unsafe_allow_html=True)
        symbol = {'lt': '<', 'le': '<=', 'eq': '==', 'ne': '!=', 'ge': '>=', 'gt': '>'}.get(parsed['relation'], '?')
        st.markdown(chips([(parsed['operation'], ''), (f'n {symbol} {parsed["threshold"]}', '')]), unsafe_allow_html=True)
    confidence = min((s['probability'] for s in result['steps']), default=0)
    weakest = min(result['steps'], key=lambda s: s['probability']) if result['steps'] else None
    st.markdown("<div class='section-label'>Confidence</div>", unsafe_allow_html=True)
    st.progress(confidence, text=f"lowest step probability {confidence:.2f}"
                + (f" (at `{weakest['token'].split(':')[-1]}`)" if weakest and confidence < 0.99 else ''))
    checks = []
    if expected_code and result['code'] and not syntax_error:
        names = argument_names(expected_code)
        key = ','.join(names)
        cases = (reference['tests'] and [t['args'] for t in reference['tests']]) if reference else None
        cases = cases or SAMPLE_ARGS.get(key, lambda k: [])(request.get('k', 0))
        checks = check_against(result['code'], expected_code, cases)
    if checks:
        passed = sum(c['ok'] for c in checks)
        status.append((f'{passed}/{len(checks)} checks pass', 'good' if passed == len(checks) else 'bad'))
    elif not expected_code:
        status.append(('could not read the intended task from the sentence: try your own input below', 'warn'))
    st.markdown("<div class='section-label'>Verdict</div>", unsafe_allow_html=True)
    st.markdown(chips(status), unsafe_allow_html=True)
    if result['unknown']:
        st.markdown(chips([(f'<UNK>: {w}', 'unk') for w in sorted(set(result['unknown']))]), unsafe_allow_html=True)

if checks:
    passed = sum(c['ok'] for c in checks)
    if passed == len(checks):
        st.success('The generated function behaves like the intended one on every check.')
    else:
        st.error('The generated function does **not** do what the sentence asks. The sentence describes:')
        st.code(expected_code, language='python')
    with st.expander(f'See the {len(checks)} checks'):
        st.dataframe(pd.DataFrame([{**c, 'ok': '✅' if c['ok'] else '❌'} for c in checks]), hide_index=True,
                     width='stretch')

if result['code'] and not syntax_error:
    with st.expander('Run it on your own input'):
        names = argument_names(result['code'])
        default = {'numbers': '[[3, 5, 5, -2, 9]]', 'text': '["Hello World"]', 'text,char': '["Banana", "a"]',
                   'a,b': '[7, 3]', 'n': '[4]'}.get(','.join(names), '[1]')
        raw = st.text_input(f"Arguments for solve({', '.join(names)}) as a JSON list", default)
        try:
            args = json.loads(raw)
            if not isinstance(args, list):
                raise ValueError('use a JSON list, e.g. [[1, 2, 3]]')
            st.markdown(f'Result: `{run_safely(result["code"], args)!r}`')
        except Unsupported as error:
            st.warning(f'Outside the subset the safe interpreter runs: {error}')
        except Exception as error:
            st.warning(f'{type(error).__name__}: {error}')

if st.session_state.pop('scroll', False):
    scroll_script = """<script>
      const go = () => { const el = window.parent.document.getElementById('result');
                         if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'}); else setTimeout(go, 100); };
      setTimeout(go, 150);
    </script>"""
    if hasattr(st, 'iframe'):            # Streamlit >= 1.5x
        st.iframe(scroll_script, height=1)
    else:                                # older Streamlit
        import streamlit.components.v1 as components
        components.html(scroll_script, height=0)

# ---------------------------------------------------------------- details
st.markdown('### How the model got there')
st.caption('Each stage of the pipeline, in order. Open the ones you are curious about.')

with st.expander('① Reading your sentence: normalization, tokens and ids'):
    st.markdown('The sentence is lowercased and split into words, numbers and punctuation (`-3` → `-`, `3`); '
                '`<EOS>` marks the end. Each token is looked up in the 97-word vocabulary learned from the training set.')
    st.markdown(chips([(f"{t} · {i}", 'unk' if t in result['unknown'] else '')
                       for t, i in zip(result['source_tokens'], result['source_ids'])]), unsafe_allow_html=True)
    if result['unknown']:
        st.caption('Red tokens are unknown and all receive id 1 (`<UNK>`).')

with st.expander('② Embeddings: every token becomes 128 numbers'):
    st.markdown('Each id selects a learned row of a 97 × 128 table. Nothing is pretrained: the values were learned '
                'while training on the dataset.')
    st.altair_chart(heatmap(result['embedded'], result['source_tokens'], 'Token', 'Embedding dimension (128)',
                            'One row per token'), width='stretch')

with st.expander('③ Encoder: reading the whole sentence'):
    if model.bidirectional:
        st.markdown('Two LSTMs read the sentence, one left → right and one right → left (128 units each). Their states '
                    'are joined, so every row below summarises the **whole** sentence.')
    else:
        st.markdown('An LSTM reads the sentence left → right; row *i* only knows the words up to *i*.')
    st.altair_chart(heatmap(result['encoder_outputs'], result['source_tokens'], 'After token', 'Hidden unit (256)',
                            'Encoder state after each token'), width='stretch')
    st.markdown('The **final state** starts the decoder.' + (
        ' With attention, at every step the decoder also re-reads every row above and feeds a weighted sum (the context vector) into its LSTM.' if model.attention else
        ' Without attention, these 256 numbers are all the decoder ever sees of your sentence.'))
    st.altair_chart(heatmap([result['final_hidden']], ['final state'], '', 'Hidden unit (256)', ''), width='stretch')

with st.expander('④ Decoder: writing the code one token at a time', expanded=False):
    st.markdown('Starting from `<SOS>`, the decoder gives a probability to each of the 51 code tokens, keeps the most '
                'likely (greedy decoding) and feeds it back in, until it writes `<EOS>`.')
    steps = result['steps']
    if steps:
        default_step = parsed['relation_index'] if parsed and parsed['relation_index'] < len(steps) else 0
        index = st.slider('Step', 0, len(steps) - 1, default_step)
        if parsed and default_step:
            st.caption(f'Opened at step {default_step}, where the comparison is written.')
        s = steps[index]
        written = ' '.join(t['token'].split(':')[-1] for t in steps[:index])
        st.markdown(f"Already written: `{written or '(nothing)'}` → next: **`{s['token'].split(':')[-1]}`** "
                    f"(probability {s['probability']:.3f})")
        cols = st.columns(2 if s['attention'] is not None else 1)
        cols[0].altair_chart(bars(pd.DataFrame(s['top'], columns=['token', 'probability']), 'probability', 'token',
                                  'Most likely next tokens', s['token']), width='stretch')
        if s['attention'] is not None:
            att = pd.DataFrame({'word': [f'{j:02d} {t}' for j, t in enumerate(result['source_tokens'])],
                                'weight': s['attention']})
            cols[1].altair_chart(bars(att, 'weight', 'word', 'Where it looks in your sentence',
                                      att.loc[att['weight'].idxmax(), 'word']), width='stretch')
        st.dataframe(pd.DataFrame([{'step': t['step'], 'wrote': t['token'], 'probability': round(t['probability'], 3),
                                    'runner-up': f"{t['top'][1][0]} ({t['top'][1][1]:.3f})" if len(t['top']) > 1 else ''}
                                   for t in steps]), hide_index=True, width='stretch')

with st.expander('⑤ Attention map: which words each code token looked at'):
    if model.attention and result['steps']:
        st.altair_chart(attention_map(result['steps'], result['source_tokens']), width='stretch')
        st.caption('Darker = more attention; each row sums to 1. Row i is computed from the previous decoder state and its '
                   'context feeds the LSTM that writes token i (Bahdanau). In our experiments the weights mostly fetch the '
                   'number; they differ a lot between runs that are all correct, so read them as a view of the '
                   'mechanism, not proof of understanding.')
    else:
        st.info('This architecture has no attention. Choose an attention model in the sidebar to see this map.')

with st.expander('⑥ Same sentence, every architecture'):
    compare_seed = st.selectbox('Training run (seed)', sorted({r['seed'] for r in runs}, key=lambda s: (s != 42, s)),
                                key='compare_seed')
    for v in VARIANT_ORDER:
        match = sorted([r for r in runs if r['variant'] == v and r['seed'] == compare_seed],
                       key=lambda r: r['device'] != 'cpu')
        if not match:
            continue
        m, voc, _ = load_model(match[0]['checkpoint'])
        out = run_pipeline(m, voc, request['instruction'], max_new_tokens, 3)
        verdict = ('invalid Python', 'bad') if (out['code'] is None or syntax_error_of(out['code'])) else None
        if verdict is None and expected_code:
            rows = check_against(out['code'], expected_code,
                                 [t['args'] for t in reference['tests']] if reference else
                                 SAMPLE_ARGS.get(','.join(argument_names(expected_code)), lambda k: [])(request.get('k', 0)))
            verdict = ('correct', 'good') if rows and all(r['ok'] for r in rows) else ('wrong', 'bad')
        st.markdown(f'**{VARIANT_LABEL[v]}** ' + chips([verdict or ('no reference', 'warn')]), unsafe_allow_html=True)
        st.code(out['code'] or ' '.join(out['output_tokens']), language='python')
