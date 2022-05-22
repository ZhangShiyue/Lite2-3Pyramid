ids.txt: the unique IDs of the 100 XSum examples

documents.txt: the source documents of the 100 XSum examples

references.txt: the gold summaries of the 100 XSum examples

summaries/: the summaries produced by 10 systems for the 100 XSum examples

SCUs.txt: the human-labeled Summary Content Units (SCUs) for the 100 XSum examples; 
each line contains SCUs for one gold summary; SCUs are separated by '\t'

labels/: the human-labeled SCU-presence of the summaries produced by 10 systems; 
each line contains SCU-presence labels (1: present, 0:not present) for one system summary; 
labels are separated by '\t'; the order of labels corresponds to the order of SCUs in SCUs.txt

STUs.txt: Automatically extracted Semantic Triplet Units (STUs) for the 100 XSum examples;
each line contains STUs for one gold summary; STUs are separated by '\t'

STUs_SCUs_percentage50.txt: The mixing of half STUs and half SCUs guided by regressors/TAC08/all_xgb.json; 
this is used to compute **Lite<sup>2.5</sup>Pyramid**.