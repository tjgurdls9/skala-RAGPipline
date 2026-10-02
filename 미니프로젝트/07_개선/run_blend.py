import sys; from blendlib import *
ex = set(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else set()
names, O, T = load(ex); print(len(names), names)
for nm, m in [('greedy', m_greedy), ('LR C=.1', m_lr(0.1)), ('LR C=.01', m_lr(0.01)), ('LGBM', m_lgbm)]:
    o, n = evaluate(O, m); print(f'{nm:9s} optimistic: {fmt(o)}\n{"":9s} nested    : {fmt(n)}', flush=True)
