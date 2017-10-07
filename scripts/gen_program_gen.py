import argparse
import os

parser = argparse.ArgumentParser(description='gen_program.sh generator')
parser.add_argument('path', type=str, help='Path(folder) to original gen_program.sh')
args = parser.parse_args()

with open(os.path.join(args.path, 'gen_program.sh')) as f:
    script = f.read().split("\n")

script[2]=''
script[3]=script[3][:-2]+'$1'
script[4]=script[4][:-2]+'$2'
script.insert(5,'enablemodel=$3')
cmd1 = script[8][:32]+'deepcoder.py predict ${example} -e ${enablemodel} '
cmd2 = script[8][-82:-62]+'$(dirname $0)/../python/output '+script[8][-62:]
del(script[-2])
script.insert(8, cmd1)
script.insert(9, cmd2)
del(script[2])

try:
    os.remove('gen_program.sh')
except OSError:
    pass
with open('gen_program.sh','w+') as f:
    for line in script:
        f.write(line +'\n')
