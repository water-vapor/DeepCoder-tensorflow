import argparse
import os

parser = argparse.ArgumentParser(description='util.py generator')
parser.add_argument('path', type=str, help='Path(folder) to original model.py and predictor.py')
args = parser.parse_args()

with open(os.path.join(args.path, 'model.py')) as m:
    model_lines = m.read().split("\n")
with open(os.path.join(args.path, 'predictor.py')) as p:
    pred_lines = p.read().split("\n")
part1 = model_lines[-45:]
to_replace = [('integer_min','params[\'integer_min\']'),
              ('integer_range','params[\'integer_range\']'),
              ('list_length','params[\'max_list_len\']'),
              ('input_num','params[\'num_input\']')]

def fold_replace(expr, li):
    for old, new in li:
        expr=expr.replace(old, new)
    return expr

rep1 = [fold_replace(l, to_replace) for l in part1]

for i in [7,24,35,42]:
    rep1[i]=rep1[i].replace('):', ', params):')
for i in [31,38,43]:
    rep1[i]=rep1[i].replace(') for', ', params) for')
rep1[32]=rep1[32].replace(')])', ', params)])')
rep1[16]=rep1[16].replace('convert_integer(x)', 'x-params[\'integer_min\']')
part2 = pred_lines[27:37]
part2[0]=part2[0].replace('):',', f):')
part2[-2]=part2[-2].replace(')',', file=f)')
imp=['import numpy as np','']
script = imp+part2+rep1

try:
    os.remove('util.py')
except OSError:
    pass
with open('util.py','w+') as f:
    for line in script:
        f.write(line +'\n')
