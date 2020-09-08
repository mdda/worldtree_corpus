# worldtree_corpus
Worldtree Corpus helper files, and sample solutions

Note: To view our submission code for the EMNLP TextGraphs 2019 Workshop, please refer to the textgraphs [branch](https://github.com/mdda/worldtree_corpus/tree/textgraphs)

The idea is to let you :

```
%load_ext autoreload
%autoreload 2
```

and

```
import os
if not os.path.isdir('worldtree_corpus'):
  ! git clone https://github.com/mdda/worldtree_corpus
import worldtree_corpus as wtc
```

at the top of a notebook, and have a bunch of useful stuff ready-to-go 
(you can choose the name under which to import it, 
so as to avoid collisions with your existing code).


### Text cleansing

```
wtc.preprocess.convert_texts(["Which of these will most likely increase?", "Habitats support animals."])
```

### Preprocessing 


```
import pandas
df_exp = wtc.preprocess.XYZ()
```

### New-style install and preprocessing

```
git clone https://github.com/mdda/worldtree_corpus.git
cd worldtree_corpus  # i.e. the REPO root directory
#git branch -a
git checkout -b textgraphs_2020 origin/textgraphs_2020
./run_setup.bash
./run_baseline.bash
. env3/bin/activate
cd src
python3 dataset.py
```
