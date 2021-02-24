# worldtree_corpus
Worldtree Corpus helper files, and sample solutions

Note: To view submission code for :

*  the EMNLP TextGraphs 2019 Workshop, please refer to the [textgraphs branch](https://github.com/mdda/worldtree_corpus/tree/textgraphs)
*  the COLING TextGraphs 2020 Workshop, please refer to the [textgraphs_2020 branch](https://github.com/mdda/worldtree_corpus/tree/textgraphs_2020)

# Base code

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
