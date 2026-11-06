"""Deterministic, standard-library-only pilot dataset. Python 3.10+.
Executes ONLY trusted reference templates authored in this file, never model output.
"""
import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
WRAPPERS = [
    'Write {signature} that will {goal}.',
    'Implement {signature} to {goal}.',
    'Create {signature}; it should {goal}.',
    'Define {signature} to {goal}.',
    'The function {signature} must {goal}.',  # validation wording
    'Using the signature {signature}, {goal}.',  # test wording
]
# These combinations NEVER appear in training, at any threshold.
HELD_OUT = {('sum', 'gt'), ('count', 'le'), ('filter', 'ne')}


def build_tasks():
    tasks = []

    def add(task_id, family, signature, goals, expression, cases, oracle,
            composition=False, structure=None):
        code = f'def {signature}:\n    return {expression}\n'
        ast.parse(code)
        tests = [{'args': args, 'expected': oracle(*args)} for args in cases]
        env = {}
        exec(compile(code, '<trusted-reference>', 'exec'), env)
        for test in tests:
            actual = env['solve'](*test['args'])
            assert type(actual) is type(test['expected']) and actual == test['expected'], task_id
        tasks.append(dict(task_id=task_id, family=family, signature=signature,
                          goals=goals, code=code, tests=tests,
                          composition=composition, structure=structure or task_id))

    binary = [
        ('add', 'a + b', ['return the sum of a and b', 'add a and b'], lambda a,b:a+b),
        ('subtract', 'a - b', ['subtract b from a', 'return a minus b'], lambda a,b:a-b),
        ('multiply', 'a * b', ['return the product of a and b', 'multiply a by b'], lambda a,b:a*b),
    ]
    for name, expr, goals, oracle in binary:
        add(name, 'arithmetic', 'solve(a, b)', goals, expr,
            [[3,4],[-2,2],[0,0],[7,-3],[-4,-5]], oracle)
    for name, expr, goals, oracle in [
        ('square','n * n',['return the square of n','multiply n by itself'],lambda n:n**2),
        ('even','n % 2 == 0',['check whether integer n is even','return whether n is divisible by 2'],lambda n:n in range(-20,21,2)),
        ('odd','n % 2 != 0',['check whether integer n is odd','return whether n is not divisible by 2'],lambda n:n not in range(-20,21,2)),
    ]:
        add(name, 'condition' if name != 'square' else 'arithmetic', 'solve(n)', goals,
            expr, [[n] for n in [-8,-3,0,1,2,9,12]], oracle)

    relations = [
        ('gt','>','greater than','above',lambda x,k:x>k),
        ('ge','>=','greater than or equal to','at least',lambda x,k:x>=k),
        ('lt','<','less than','below',lambda x,k:x<k),
        ('le','<=','less than or equal to','at most',lambda x,k:x<=k),
        ('eq','==','equal to','exactly',lambda x,k:x==k),
        ('ne','!=','not equal to','different from',lambda x,k:x!=k),
    ]
    for relation, symbol, phrase, synonym, predicate in relations:
        for k in range(-5, 11):
            for operation in ['filter','count','sum']:
                descriptions = {'filter': 'return, in their original order and retaining duplicates, the values',
                                'count': 'count the values', 'sum': 'return the sum of the values'}
                goals = [f'{descriptions[operation]} in numbers that are {p} {k}'
                         for p in [phrase,synonym]]
                inner = f'n for n in numbers if n {symbol} {k}'
                expression = {'filter':f'[{inner}]', 'sum':f'sum({inner})',
                              'count':f'sum(1 for n in numbers if n {symbol} {k})'}[operation]
                cases = [[[]], [[k]], [[k-1,k,k+1]], [[k+1,k+1,k-1]], [[-9,0,13]]]
                def oracle(xs, pred=predicate, threshold=k, op=operation):
                    selected = []
                    for value in xs:
                        if pred(value, threshold):
                            selected.append(value)
                    if op == 'filter':
                        return selected
                    if op == 'count':
                        return len(selected)
                    total = 0
                    for value in selected:
                        total += value
                    return total
                add(f'{operation}_{relation}_{k}', 'lists', 'solve(numbers)', goals,
                    expression, cases, oracle, (operation,relation) in HELD_OUT,
                    f'{operation}_{relation}')

    for name, expr, goals, oracle in [
        ('reverse','text[::-1]',['reverse the string text','return text in reverse order'],lambda t:''.join(reversed(t))),
        ('lower','text.lower()',['convert text to lowercase','return a lowercase version of text'],lambda t:t.lower()),
        ('upper','text.upper()',['convert text to uppercase','return an uppercase version of text'],lambda t:t.upper()),
    ]:
        add(name, 'strings', 'solve(text)', goals, expr,
            [[t] for t in ['', 'a','Hello World','AbBa','123!?']], oracle)
    add('char_count','strings','solve(text, char)',
        ['count occurrences of the single character char in text, respecting case',
         'return how many times the single character char appears in text, case-sensitively'],
        'text.count(char)', [['Banana','a'],['','x'],['AaA','a'],['aaa','a'],['abc','z']],
        lambda t,c:len([x for x in t if x==c]))
    return tasks


def main():
    root = Path(__file__).resolve().parent / 'data'
    root.mkdir(exist_ok=True)
    tasks = build_tasks()
    splits = {name:[] for name in ['train','validation_wording','test_wording','test_composition']}
    for task in tasks:
        for wording_id, wrapper in enumerate(WRAPPERS):
            for variant, goal in enumerate(task['goals']):
                # Composition uses training-style wording to isolate novel combinations.
                if task['composition']:
                    if wording_id != 0:
                        continue
                    split = 'test_composition'
                else:
                    split = 'train' if wording_id < 4 else 'validation_wording' if wording_id == 4 else 'test_wording'
                row = {k:task[k] for k in ['task_id','family','structure','code','tests']}
                row.update(id=f"{task['task_id']}_w{wording_id}_v{variant}",
                           instruction=wrapper.format(signature=task['signature'],goal=goal),
                           wording_id=wording_id)
                splits[split].append(row)
    # Instructions must never repeat across splits.
    all_inputs = [r['instruction'] for rows in splits.values() for r in rows]
    assert len(all_inputs) == len(set(all_inputs))
    train_tasks = {r['task_id'] for r in splits['train']}
    assert train_tasks == {r['task_id'] for r in splits['test_wording']}
    assert train_tasks.isdisjoint(r['task_id'] for r in splits['test_composition'])
    train_structures = {r['structure'] for r in splits['train']}
    assert train_structures.isdisjoint(r['structure'] for r in splits['test_composition'])
    train_codes = {r['code'] for r in splits['train']}
    assert train_codes.isdisjoint(r['code'] for r in splits['test_composition'])
    summary = {'seed':SEED, 'unique_tasks':len(tasks),
               'unique_program_structures':len({t['structure'] for t in tasks}),
               'reference_test_checks':sum(len(t['tests']) for t in tasks),
               'held_out_compositions':sorted([list(x) for x in HELD_OUT]), 'splits':{}}
    for name, rows in splits.items():
        random.Random(SEED).shuffle(rows)
        target = root / f'{name}.jsonl'
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        summary['splits'][name] = {'examples':len(rows),
             'unique_tasks':len({r['task_id'] for r in rows}),
             'families':dict(Counter(r['family'] for r in rows)),
             'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
