# Lite<sup>2-3</sup>Pyramid (EMNLP 2021)

This repository contains the code and data for the following paper:

[Finding a Balanced Degree of Automation for Summary Evaluation]()

```
@inproceedings{zhang2021finding,
  title={Finding a Balanced Degree of Automation for Summary Evaluation},
  author={Zhang, Shiyue and Bansal, Mohit},
  booktitle={The 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021}
}
```

### Requirements

* Python 3
* requirements.txt

### Quick Start
Run the following command to get the Lite<sup>2</sup>Pyramid score for abstractive BART summaries 
for 100 CNN/DM examples from [REALSumm](https://github.com/neulab/REALSumm). 

```
python Lite2-3Pyramid.py --units data/REALSumm/SCUs.txt --summaries data/REALSumm/abs_bart_out.summary --model shiyue/roberta-large-tac08

expected output: {'p2c': 0.40807125, 'l2c': 0.42, 'p3c': 0.35005152, 'l3c': 0.42}
```

Usually, "p2c" should be token as the final score.
When using the zero-shot pretrained model (i.e., ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli), 
"l3c" should be used. 

When Summary Content Units (SCUs) or Semantic Triplet Units (STUs) are ready, simply input 
the "unit file" that stores SCUs or STUs (or SCUs+STUs) via --units, 
and input the "summary file" that stores system summaries via --summaries.  
Each line of the summary file is one summary.
Each line of the unit file has the units for one reference, and units are separated by '\t'.
The unit file and the summary file need to be aligned.

### Pretrained NLI Models
We provide 4 NLI models finetuned on the 4 meta-evaluation sets respectively. 

We suggest using X-finetuned NLI model on X dataset. When evaluating on a new dataset, 
we suggest using TAC08-finetuned model (i.e., shiyue/roberta-large-tac08) by default.

| trained or finutuned on | Huggingface hub name|
| ------------- | ----------- |
|  [SNLI](https://nlp.stanford.edu/projects/snli/)+[MNLI](https://cims.nyu.edu/~sbowman/multinli/)+[FEVER](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md)+[ANLI](https://github.com/facebookresearch/anli)  | ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli |
|  [TAC2008](https://tac.nist.gov/2008/summarization/update.summ.08.guidelines.html)  | shiyue/roberta-large-tac08 |
|  [TAC2009](https://tac.nist.gov/2009/Summarization/update.summ.09.guidelines.html)  | shiyue/roberta-large-tac09 |
|  [REALSumm](https://github.com/neulab/REALSumm)  | shiyue/roberta-large-realsumm |
|  PyrXSum  | shiyue/roberta-large-pyrxsum |


### Todos
* migrate to pypi package
* pyrxsum data
* reproduction code for the paper
* provide support through [sacrerouge](https://github.com/danieldeutsch/sacrerouge)





