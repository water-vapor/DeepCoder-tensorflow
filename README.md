DeepCoder-tensorflow
===

This repository contains a model implementation for DeepCoder in tensorflow.
I did not implement the DSL, because it is too time consuming, instead, I used
the implementation from [this repo](https://github.com/HiroakiMikami/deep-coder).

Unfortunately, the original author did not include a license. To prevent copyright infringement,
I did not fork that repo and modify directly on github. Instead, I publish the code needed to assemble
a working version.

Building
---
Please refer to the guide in the original repo. You may need to compile `gtest` and the program with option
`` -DCMAKE_CXX_COMPILER=`which g++-6` -DCMAKE_C_COMPILER=`which gcc-6` `` for cmake. Make sure the code is
working before proceeding to the next step.

Generating Code for tensorflow
---
Download the source code from the original repo: [this commit](https://codeload.github.com/HiroakiMikami/deep-coder/zip/b11a07d4d2113f69d2ea69015c35db18879e7758).
There are two python scripts that will generate the appropriate files from the original source code.
Navigate to `DeepCoder-tensorflow` folder. `deep-coder-master` refers to the folder of the original repo.
```bash
cd scripts
python3 gen_program_gen.py path/to/deep-coder-master/scripts/
cd ..
cd python
python3 util_gen.py path/to/deep-coder-master/python/
```
Now copy all the files in `DeepCoder-tensorflow` to `deep-coder-master`, replace the conflicting files.

Usage
---
Search with the help of the model
```bash
$ time ./scripts/gen_program.sh examples/1.json 3 true dfs

head    last    take    drop    access  minimum maximum reverse sort    sum map filter  count   zip_with    scanl1  >0  <0  %2 == 0 %2 == 1+1   -1  *(-1)   *2  *3  *4  /2  /3  /4  **2 MIN MAX
0.0831  0.0779  0.254   0.0722  0.00302 0.0041  0.0923  0.706   0.273   0.009720.059    0.185   0.154   0.0288  0.035   0.0107  0.0932  0.000624    0.002110.0053   0.00189 0.057   0.00132 0.00355 0.00775 0.000167    0.0171  0.194   0.127   0.0298  0.334   0.111   4.07e-11    4.19e-11    4.19e-11
---
a <- read_list
b <- read_list
c <- sort a
d <- map **2 c
e <- zip_with MIN d b
---


real    0m18.947s
user    0m18.621s
sys 0m0.275s
```
Search without the model
```bash
$ time ./scripts/gen_program.sh examples/1.json 3 false none

---
a <- read_list
b <- read_list
c <- sort a
d <- map **2 c
e <- zip_with MIN d b
---


real    0m31.580s
user    0m31.093s
sys 0m0.307s

```

Training
---
```bash
$ python3 ./python/deepcoder.py train dataset/dataset.json
```
