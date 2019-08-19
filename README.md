# worldtree_corpus
Worldtree Corpus helper files, and sample solutions

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
