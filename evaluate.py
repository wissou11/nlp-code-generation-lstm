"""Evaluate a frozen model (baseline or attention) on both test sets.

python evaluate.py --checkpoint runs/BASELINE_RUN/checkpoint_epoch_0030.pt
python evaluate.py --checkpoint runs/ATTENTION_RUN/checkpoint_epoch_00NN.pt

Teacher-forced token accuracy and perplexity (exp of mean token cross-entropy,
reference prefix fed to the decoder) are reported as diagnostics; they measure
next-token prediction, not whether the generated program is correct.
Besides syntax/behavioral metrics, list tasks get CONDITION-RETENTION metrics
read from the generated tokens: is the operation (filter/count/sum), the
relation operator (<, <=, ==, !=, >=, >) and the threshold the requested one?
These say WHERE a wrong program goes wrong. For attention checkpoints, the
source tokens most attended while emitting the relation operator are saved.

Generated code is NEVER passed to exec/eval. Behavioral checks use a bounded
AST interpreter supporting only the pilot's expression-based Python subset.
Unsupported syntax is counted separately and fails behavioral checks. This is
restricted-subset behavioral evaluation, not unrestricted Python execution.
Test results must not be used to tune the model. Defaults use greedy decoding.
"""
import argparse
import ast
import hashlib
import json
import math
import operator
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import torch
from model import Seq2Seq, collate_batch, EOS_ID
from preprocess import tokenize_instruction, tokenize_code, encode, detokenize_code


class Unsupported(ValueError):
    pass


class RestrictedPython:
    """Bounded interpretation of one function containing one return expression.

    Supports numbers, strings, lists, comprehensions, comparisons, arithmetic,
    slicing, sum and selected string methods. No imports, file/network access,
    arbitrary calls, recursion, assignments, loops or dynamic attribute access.
    This is a domain-specific interpreter, not a general Python sandbox.
    """
    OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Mod: operator.mod}
    CMP = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}
    ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
               ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.UnaryOp, ast.USub,
               ast.UAdd, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
               ast.Eq, ast.NotEq, ast.Call, ast.Attribute, ast.Subscript,
               ast.Slice, ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp,
               ast.comprehension)

    def __init__(self, code):
        if len(code) > 10000:
            raise Unsupported('Code length limit')
        tree = ast.parse(code)
        nodes = list(ast.walk(tree))
        if len(nodes) > 250 or any(not isinstance(n, self.ALLOWED) for n in nodes):
            raise Unsupported('Unsupported node or node limit')
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise Unsupported('Expected one function')
        self.fn = fn = tree.body[0]
        args = fn.args
        if (fn.name != 'solve' or fn.decorator_list or fn.returns or
            getattr(fn, 'type_params', []) or args.posonlyargs or args.vararg or
            args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults or
            any(a.annotation for a in args.args) or len(fn.body) != 1 or
            not isinstance(fn.body[0], ast.Return)):
            raise Unsupported('Expected plain solve(...) with one return')
        self.names = [a.arg for a in args.args]
        if len(set(self.names)) != len(self.names):
            raise Unsupported('Duplicate argument names')
        # Generator expressions are materialized internally only as sum arguments.
        # Returning a generator must not be mistaken for returning a Python list.
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
        for node in nodes:
            if isinstance(node, ast.GeneratorExp):
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                        and parent.func.id == 'sum' and parent.args == [node]):
                    raise Unsupported('Generator expression supported only directly inside sum')
        # Validate every call, including calls in branches not reached by a test.
        for node in nodes:
            if isinstance(node, ast.Call):
                valid = ((isinstance(node.func, ast.Name) and node.func.id == 'sum') or
                         (isinstance(node.func, ast.Attribute) and node.func.attr in
                          {'lower', 'upper', 'count'}))
                if not valid or node.keywords:
                    raise Unsupported('Unsupported function call')
            if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
                if (len(node.generators) != 1 or node.generators[0].is_async or
                    not isinstance(node.generators[0].target, ast.Name)):
                    raise Unsupported('Only one simple comprehension is supported')
        self.remaining = 0

    def bounded(self, value):
        if type(value) in (int, float):
            if abs(value) > 1000000:
                raise Unsupported('Numeric limit')
        elif isinstance(value, (str, list, tuple)):
            if len(value) > 200:
                raise Unsupported('Collection size limit')
            if isinstance(value, (list, tuple)):
                for child in value:
                    self.bounded(child)
        elif value is not None and type(value) is not bool:
            raise Unsupported('Unsupported value type')
        return value

    def run(self, args):
        if len(args) != len(self.names):
            raise TypeError('Argument count mismatch')
        self.remaining = 2000
        for arg in args:
            self.bounded(arg)
        return self.expression(self.fn.body[0].value, dict(zip(self.names,args)))

    def expression(self, node, env):
        self.remaining -= 1
        if self.remaining < 0:
            raise Unsupported('Operation budget exceeded')
        if isinstance(node, ast.Constant):
            return self.bounded(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.expression(n,env) for n in node.elts]
            return self.bounded(tuple(values) if isinstance(node,ast.Tuple) else values)
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand,env)
            if type(value) not in (int,float):
                raise Unsupported('Unary operand must be numeric')
            return self.bounded(-value if isinstance(node.op,ast.USub) else +value)
        if isinstance(node, ast.BinOp):
            left,right = self.expression(node.left,env),self.expression(node.right,env)
            # Numeric arithmetic only; blocks huge string/list multiplication.
            if type(left) not in (int,float) or type(right) not in (int,float):
                raise Unsupported('Arithmetic operands must be numeric')
            return self.bounded(self.OPS[type(node.op)](left,right))
        if isinstance(node, ast.Compare):
            left = self.expression(node.left,env)
            for op, operand in zip(node.ops,node.comparators):
                right = self.expression(operand,env)
                if not self.CMP[type(op)](left,right):
                    return False
                left = right
            return True
        if isinstance(node, (ast.ListComp,ast.GeneratorExp)):
            generator = node.generators[0]
            iterable = self.expression(generator.iter,env)
            if not isinstance(iterable,(list,tuple,str)):
                raise Unsupported('Unsupported iterable')
            values = []
            for item in iterable:
                local = dict(env)
                local[generator.target.id] = item
                if all(self.expression(c,local) for c in generator.ifs):
                    values.append(self.expression(node.elt,local))
            return self.bounded(values)
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value,env)
            if not isinstance(value,(str,list,tuple)):
                raise Unsupported('Unsupported subscript target')
            if isinstance(node.slice,ast.Slice):
                parts=[self.expression(x,env) if x is not None else None
                       for x in [node.slice.lower,node.slice.upper,node.slice.step]]
                index=slice(*parts)
            else:
                index=self.expression(node.slice,env)
            return self.bounded(value[index])
        if isinstance(node, ast.Call):
            args=[self.expression(x,env) for x in node.args]
            if isinstance(node.func,ast.Name):
                if node.func.id in env or len(args) != 1 or not isinstance(args[0],(list,tuple)):
                    raise Unsupported('Expected unshadowed sum of one collection')
                if any(type(x) not in (int,float,bool) for x in args[0]):
                    raise Unsupported('Expected numeric sum')
                return self.bounded(sum(args[0]))
            value=self.expression(node.func.value,env)
            if not isinstance(value,str):
                raise Unsupported('Only string methods supported')
            method=node.func.attr
            if method=='lower' and not args:
                return self.bounded(value.lower())
            if method=='upper' and not args:
                return self.bounded(value.upper())
            if method=='count' and len(args)==1 and isinstance(args[0],str):
                return value.count(args[0])
            raise Unsupported('Unsupported method arguments')
        raise Unsupported('Unsupported expression')


RELATIONS = {'OP:<': 'lt', 'OP:<=': 'le', 'OP:==': 'eq', 'OP:!=': 'ne',
             'OP:>=': 'ge', 'OP:>': 'gt'}


def parse_condition(tokens):
    """Read operation, relation and threshold from list-task code tokens.

    Returns None when the tokens do not follow the list-task shape
    'return <[n for n in numbers | sum(1 for ... | sum(n for ...> if n REL THR'.
    Also returns the index of the relation token (for attention diagnostics).
    """
    try:
        r = tokens.index('NAME:return')
        i = tokens.index('NAME:if', r)
    except ValueError:
        return None
    head = tokens[r+1:r+4]
    if head[:1] == ['OP:[']:
        operation = 'filter'
    elif head[:2] == ['NAME:sum', 'OP:('] and head[2:] == ['NUMBER:1']:
        operation = 'count'
    elif head[:2] == ['NAME:sum', 'OP:('] and head[2:] == ['NAME:n']:
        operation = 'sum'
    else:
        operation = 'other'
    if tokens[i+1:i+2] != ['NAME:n'] or len(tokens) <= i + 2:
        return None
    relation = RELATIONS.get(tokens[i+2], 'other')
    threshold = []
    for item in tokens[i+3:]:
        if item in ('OP:)', 'OP:]'):
            break
        threshold.append(item.partition(':')[2])
    return {'operation': operation, 'relation': relation,
            'threshold': ''.join(threshold), 'relation_index': i + 2}


def condition_record(generated_tokens, reference_tokens):
    reference = parse_condition(reference_tokens)
    if reference is None:
        return {}
    generated = parse_condition(generated_tokens) or {
        'operation': 'unparsed', 'relation': 'unparsed', 'threshold': None}
    return {'requested_operation': reference['operation'],
            'requested_relation': reference['relation'],
            'requested_threshold': reference['threshold'],
            'generated_operation': generated['operation'],
            'generated_relation': generated['relation'],
            'generated_threshold': generated['threshold'],
            'operation_correct': generated['operation'] == reference['operation'],
            'relation_correct': generated['relation'] == reference['relation'],
            'threshold_correct': generated['threshold'] == reference['threshold']}


def same_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual,list):
        return len(actual)==len(expected) and all(same_value(a,b) for a,b in zip(actual,expected))
    return actual==expected


def assess(code, tests, ended):
    result={'syntax_valid':False,'subset_supported':False,
            'all_tests_passed':False,'passed_tests':0,'total_tests':len(tests),'failure':None}
    if code is None:
        result['failure']='Cannot reconstruct code tokens'
        return result
    try:
        # Compile checks Python syntax/context; the code object is never run.
        compile(code,'<generated>','exec')
        result['syntax_valid']=True
    except (SyntaxError,ValueError,TypeError) as error:
        result['failure']=f'Syntax error: {error}'
        return result
    try:
        interpreter=RestrictedPython(code)
        result['subset_supported']=True
        for test in tests:
            actual=interpreter.run(test['args'])
            if same_value(actual,test['expected']):
                result['passed_tests']+=1
            elif result['failure'] is None:
                result['failure']=f"Input {test['args']!r}: expected {test['expected']!r}, got {actual!r}"
        result['all_tests_passed']=ended and result['passed_tests']==len(tests)
        if not ended:
            result['failure']='Generation reached length limit without EOS'
    except Unsupported as error:
        result['subset_supported']=False
        result['failure']=f'Outside supported subset: {error}'
    except (ArithmeticError,TypeError,ValueError,KeyError,IndexError,RecursionError) as error:
        result['failure']=f'Behavioral error: {type(error).__name__}: {error}'
    return result


def corpus_bleu(candidates, references, max_n=4):
    """BLEU-4 over the whole split (Papineni et al. 2002), implemented here.

    p_n = clipped matching n-grams / n-grams in the generated sequences, for
    n = 1..4; BLEU = brevity penalty x (p_1 p_2 p_3 p_4)^(1/4), the geometric
    mean of the n-gram precisions as presented in the course. Computed on code
    tokens (layout tokens included). Returns a value in [0, 1].
    """
    from collections import Counter
    matches = [0] * max_n
    totals = [0] * max_n
    candidate_length = reference_length = 0
    for candidate, reference in zip(candidates, references):
        candidate_length += len(candidate)
        reference_length += len(reference)
        for n in range(1, max_n + 1):
            cand = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1))
            ref = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))
            matches[n-1] += sum(min(count, ref[gram]) for gram, count in cand.items())
            totals[n-1] += max(len(candidate) - n + 1, 0)
    if min(matches) == 0 or candidate_length == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
    brevity = 1.0 if candidate_length > reference_length else math.exp(1 - reference_length / candidate_length)
    return brevity * math.exp(log_precision)


def summarize(records):
    n=len(records)
    groups=defaultdict(list)
    for r in records:
        groups[r['task_id']].append(r)
    return {'examples':n,'unique_tasks':len(groups),
            'exact_token_match_rate':sum(r['exact_token_match'] for r in records)/n,
            'syntax_validity_rate':sum(r['syntax_valid'] for r in records)/n,
            'supported_subset_rate':sum(r['subset_supported'] for r in records)/n,
            'all_tests_pass_rate':sum(r['all_tests_passed'] for r in records)/n,
            'eos_rate':sum(r['ended_with_eos'] for r in records)/n,
            'tasks_passing_all_phrasings_rate':sum(all(r['all_tests_passed'] for r in rows)
                                                   for rows in groups.values())/len(groups),
            'bleu4':corpus_bleu([r['generated_tokens'] for r in records],
                                [tokenize_code(r['reference_code']) for r in records]),
            **condition_summary(records)}


def condition_summary(records):
    """Condition-retention rates over list-task records (others are skipped)."""
    listed = [r for r in records if 'relation_correct' in r]
    if not listed:
        return {}
    m = len(listed)
    confusion = defaultdict(lambda: defaultdict(int))
    for r in listed:
        confusion[r['requested_relation']][r['generated_relation']] += 1
    return {'condition_examples': m,
            'operation_correct_rate': sum(r['operation_correct'] for r in listed)/m,
            'relation_correct_rate': sum(r['relation_correct'] for r in listed)/m,
            'threshold_correct_rate': sum(r['threshold_correct'] for r in listed)/m,
            'relation_confusion': {k: dict(v) for k, v in sorted(confusion.items())}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error('Batch size must be positive')
    root=Path(__file__).resolve().parent
    path=Path(args.checkpoint).resolve()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(torch.get_num_threads(),4))
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    vocab=checkpoint['vocabularies']
    current=json.loads((root/'processed/vocabularies.json').read_text(encoding='utf-8'))
    if current!=vocab:
        raise ValueError('Local vocabulary does not match checkpoint; keep original preprocessing files.')
    model=Seq2Seq(**checkpoint['model_config']).to(device)
    variant=('bi_' if model.bidirectional else '')+('attention' if model.attention else 'baseline')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    out=path.parent/f'evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir()
    summary={'checkpoint':str(path),'variant':variant,
             'trainable_parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'device':str(device),'decoding':'greedy; max 64 tokens; PAD/SOS/UNK suppressed',
             'behavioral_method':'Bounded AST interpreter for pilot subset, not arbitrary Python execution',
             'test_data_hashes':{},'splits':{}}
    print(f'Device: {device}\nCheckpoint: {path}\nVariant: {variant}',flush=True)
    for split in ['test_wording','test_composition']:
        data_path=root/'data'/f'{split}.jsonl'
        rows=[json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty split: {split}')
        summary['test_data_hashes'][split]=hashlib.sha256(data_path.read_bytes()).hexdigest()
        # Check interpreter support against every reference before scoring any model outputs.
        for row in rows:
            if not assess(row['code'],row['tests'],True)['all_tests_passed']:
                raise RuntimeError(f"Reference not supported or tests inconsistent: {row['id']}")
        records=[]
        # Teacher-forced diagnostics: the decoder sees the REFERENCE prefix.
        tf_loss=tf_correct=tf_tokens=0.0
        for start in range(0,len(rows),args.batch_size):
            chunk=rows[start:start+args.batch_size]
            encoded=[]
            for row in chunk:
                target=encode(tokenize_code(row['code']),vocab['target_token_to_id'])
                encoded.append({'source_ids':encode(tokenize_instruction(row['instruction']),vocab['source_token_to_id'])+[EOS_ID],
                                'decoder_input_ids':[2]+target,'decoder_target_ids':target+[EOS_ID]})
            batch=collate_batch(encoded)
            with torch.no_grad():
                logits=model(batch['source_ids'].to(device),batch['source_lengths'],batch['decoder_input_ids'].to(device))
                labels=batch['decoder_target_ids'].to(device)
                mask=labels.ne(0)
                tf_loss+=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1),
                                                           ignore_index=0,reduction='sum').item()
                tf_correct+=(logits.argmax(-1).eq(labels)&mask).sum().item()
                tf_tokens+=mask.sum().item()
            if model.attention:
                generated,flags,attention_rows=model.generate(batch['source_ids'].to(device),batch['source_lengths'],
                                                              max_new_tokens=64,return_attention=True)
            else:
                generated,flags=model.generate(batch['source_ids'].to(device),batch['source_lengths'],max_new_tokens=64)
                attention_rows=[None]*len(chunk)
            for row,ids,ended,attention in zip(chunk,generated,flags,attention_rows):
                tokens=[vocab['target_id_to_token'][i] for i in ids]
                try:
                    code=detokenize_code(tokens)
                except (ValueError,IndexError,IndentationError,SyntaxError):
                    code=None
                condition=condition_record(tokens,tokenize_code(row['code']))
                if attention is not None and condition:
                    parsed=parse_condition(tokens)
                    if parsed is not None:
                        source_tokens=tokenize_instruction(row['instruction'])+['<EOS>']
                        weights=attention[parsed['relation_index']]
                        top=sorted(range(len(weights)),key=lambda k:-weights[k])[:3]
                        condition['relation_step_top_attention']=[[source_tokens[k],round(weights[k],3)] for k in top]
                records.append({'id':row['id'],'task_id':row['task_id'],'family':row['family'],
                    'structure':row['structure'],'instruction':row['instruction'],
                    'reference_code':row['code'],'generated_code':code,'generated_tokens':tokens,
                    'ended_with_eos':ended,'exact_token_match':ended and tokens==tokenize_code(row['code']),
                    **condition,**assess(code,row['tests'],ended)})
        metrics=summarize(records)
        # Token accuracy and perplexity include the EOS prediction, exclude PAD.
        metrics['teacher_forced_token_accuracy']=tf_correct/tf_tokens
        metrics['teacher_forced_loss']=tf_loss/tf_tokens
        metrics['perplexity']=math.exp(tf_loss/tf_tokens)
        metrics['by_family']={family:summarize([r for r in records if r['family']==family])
                              for family in sorted({r['family'] for r in records})}
        metrics['by_structure']={structure:summarize([r for r in records if r['structure']==structure])
                              for structure in sorted({r['structure'] for r in records})}
        summary['splits'][split]=metrics
        with (out/f'{split}_predictions.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for record in records:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f'\n{split}: {len(records)} examples',flush=True)
        print(f"  teacher_forced_token_accuracy: {metrics['teacher_forced_token_accuracy']:.2%}"
              f" | perplexity: {metrics['perplexity']:.4f}",flush=True)
        print(f"  BLEU-4 (code tokens): {metrics['bleu4']*100:.1f}",flush=True)
        for key in ['exact_token_match_rate','syntax_validity_rate','supported_subset_rate','all_tests_pass_rate','eos_rate',
                    'operation_correct_rate','relation_correct_rate','threshold_correct_rate']:
            if key in metrics:
                print(f'  {key}: {metrics[key]:.1%}',flush=True)
        for structure,m in metrics['by_structure'].items():
            if split=='test_composition':
                print(f"  {structure}: pass {m['all_tests_pass_rate']:.1%} | relation {m.get('relation_correct_rate',0):.1%}"
                      f" | generated relations {m.get('relation_confusion',{})}",flush=True)
        failures=[r for r in records if not r['all_tests_passed']]
        for r in failures[:3]:
            print(f"  Failure {r['id']}: {r['failure']}",flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'\nSaved summary and per-example predictions to: {out}')
    print('These are final test results. Do not tune hyperparameters on these sets.')


if __name__=='__main__':
    main()
