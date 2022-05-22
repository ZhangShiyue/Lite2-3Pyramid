# Lite<sup>2-3</sup>Pyramid (EMNLP 2021)

This repository contains the code and data for the following paper:

[Finding a Balanced Degree of Automation for Summary Evaluation](https://arxiv.org/abs/2109.11503)

```
@inproceedings{zhang2021finding,
  title={Finding a Balanced Degree of Automation for Summary Evaluation},
  author={Zhang, Shiyue and Bansal, Mohit},
  booktitle={The 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021}
}
```
Note that this repo is still **work-in-progress**.

# 

### Requirements

* Python 3
* requirements.txt

# 

### Quick Start

#### REALSumm
Run the following command to get the **Lite<sup>2</sup>Pyramid** score for abstractive BART summaries 
for 100 CNN/DM examples from [REALSumm](https://github.com/neulab/REALSumm). 
```
python Lite2-3Pyramid.py --unit data/REALSumm/SCUs.txt --summary data/REALSumm/summaries/abs_bart_out.summary --label data/REALSumm/labels/abs_bart_out.label --device 0

expected output: {'p2c': 0.5014365067350771, 'l2c': 0.5159722360972361, 'p3c': 0.43105412152833117, 'l3c': 0.4964144744144744, 'human': 0.48349483849483854, 'model_type': 'shiyue/roberta-large-tac08'}
```

To get its **Lite<sup>3</sup>Pyramid** score:
```
python Lite2-3Pyramid.py --unit data/REALSumm/STUs.txt --summary data/REALSumm/summaries/abs_bart_out.summary --device 0

expected output: {'p2c': 0.4535250155726269, 'l2c': 0.48114382512911924, 'p3c': 0.38368004398714206, 'l3c': 0.45765291326320745, 'human': None, 'model': 'shiyue/roberta-large-tac08'}
```

Usually, "p2c" should be taken as the final score.
When using the zero-shot NLI model (i.e., ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli), 
"l3c" should be used. 

**To extract STUs**:
```
python Lite2-3Pyramid.py --extract_stus --reference data/REALSumm/references.txt --doc_id data/REALSumm/ids.txt --output_dir data/REALSumm --use_coref --device 0
```
Then, the extracted STUs for REALSumm references will be saved in "data/REALSumm/STUs.txt". 
Besides, two intermediate files (ref_srls.pkl and ref_corefs.pkl) will also be saved under "data/REALSumm".

To get its **Lite<sup>2.5</sup>Pyramid** score (using the regressor trained on TAC2008):
```
python Lite2-3Pyramid.py --unit data/REALSumm/STUs_SCUs_percentage50.txt --summary data/REALSumm/summaries/abs_bart_out.summary --device 0

expected output: {'p2c': 0.47894588174027, 'l2c': 0.4990391414141414, 'p3c': 0.40794726118628005, 'l3c': 0.478387445887446, 'human': None, 'model': 'shiyue/roberta-large-tac08'}
```

**To mix STUs and SCUs**, SCUs need to be first obtained. Then, the percentage of STUs need to be specified, e.g., if 
it is 50, then there will be 50% STUs + 50% SCUs. Besides, we also need to specify the regressor used for mixing STUs and SCUs.
```
python Lite2-3Pyramid.py --mix_stus_and_scus --stu_percentage 50 --scus_file data/REALSumm/SCUs.txt --regressor regressors/TAC08/all_xgb.json --reference data/REALSumm/references.txt --doc_id data/REALSumm/ids.txt --output_dir data/REALSumm --use_coref --device 0
```
If you have intermediate SRL and Coreference results saved, you can use the following command to save time.
```
python Lite2-3Pyramid.py --mix_stus_and_scus --stu_percentage 50 --scus_file data/REALSumm/SCUs.txt --regressor regressors/TAC08/all_xgb.json --reference data/REALSumm/references.txt --srl_file data/REALSumm/ref_srls.pkl --coref_file data/REALSumm/ref_corefs.pkl --doc_id data/REALSumm/ids.txt --output_dir data/REALSumm --use_coref --device 0
```


#### PyrXSum

Similar to REALSumm, run the following command to get the **Lite<sup>2</sup>Pyramid** score for abstractive T5-large summaries 
for 100 XSum examples.
```
python Lite2-3Pyramid.py --unit data/PyrXSum/SCUs.txt --summary data/PyrXSum/summaries/t5-large.summary --label data/PyrXSum/labels/t5-large.label --device 0

expected output: {'p2c': 0.354199199620129, 'l2c': 0.37358008658008657, 'p3c': 0.27956780085637617, 'l3c': 0.32769913419913416, 'human': 0.29117532467532464, 'model': 'shiyue/roberta-large-tac08'}
```
When set "--unit data/PyrXSum/STUs.txt", it will give the **Lite<sup>3</sup>Pyramid** score.  


Similar to REALSumm (except, do not use --use_coref), **To extract STUs**:
```
python Lite2-3Pyramid.py --extract_stus --reference data/PyrXSum/references.txt --doc_id data/PyrXSum/ids.txt --output_dir data/PyrXSum/ --device 0
```
One intermediate file (ref_srls.pkl) will also be saved under "data/PyrXSum".


**To mix STUs and SCUs**,
```
python Lite2-3Pyramid.py --mix_stus_and_scus --stu_percentage 50 --scus_file data/PyrXSum/SCUs.txt --regressor regressors/TAC08/all_xgb.json --reference data/PyrXSum/references.txt --srl_file data/PyrXSum/ref_srls.pkl --doc_id data/PyrXSum/ids.txt --output_dir data/PyrXSum/ --device 0
```

### Pretrained NLI Models
We provide the zero-shot NLI model and 4 other NLI models 
finetuned on the 4 meta-evaluation sets respectively. 

We suggest using X-finetuned NLI model on X dataset. When evaluating on a new dataset, 
we suggest using TAC08-finetuned model (i.e., shiyue/roberta-large-tac08) by default.

| pretrained or finutuned on | Huggingface hub name|
| ------------- | ----------- |
|  [SNLI](https://nlp.stanford.edu/projects/snli/)+[MNLI](https://cims.nyu.edu/~sbowman/multinli/)+[FEVER](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md)+[ANLI](https://github.com/facebookresearch/anli)  | ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli |
|  [TAC2008](https://tac.nist.gov/2008/summarization/update.summ.08.guidelines.html)  | shiyue/roberta-large-tac08 |
|  [TAC2009](https://tac.nist.gov/2009/Summarization/update.summ.09.guidelines.html)  | shiyue/roberta-large-tac09 |
|  [REALSumm](https://github.com/neulab/REALSumm)  | shiyue/roberta-large-realsumm |
|  PyrXSum  | shiyue/roberta-large-pyrxsum |

#

### Reproduction
See [reproduce](reproduce)

#

### Todos
* reproduction code for cross-validation experiments on TAC08/09/PyrXSum
* reproduction code for out-of-the-box experiments
* provide version control via pypi package
* provide support through [sacrerouge](https://github.com/danieldeutsch/sacrerouge)





