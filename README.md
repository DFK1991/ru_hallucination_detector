# Detection of simple hallucinations in Russian texts using lightweight BERT-based classifier

## Overview

This project focuses on detecting simple hallucinations in Russian text using lightweight BERT-based models. The task is formulated as a text-pair classification problem, where a model determines whether an evidence sentence is supported by a given context.

The following hallucination types are considered:

* **Clean** — fully supported statement
* **Number Flip** — numerical inconsistency
* **Entity Swap** — incorrect entity substitution
* **Negation Flip** — meaning altered by negation

---

## Model

The model is based on:

* `cointegrated/rubert-tiny2` (lightweight Russian BERT)

Key components:

* Choice of a pooling strategies (attention / mean / max)
* Context–evidence interaction features (difference, similarity)
* Additional numeric features for detecting number inconsistencies

---

## Project Structure

```
project/
├── data/                # dataset (.parquet, not tracked by git)
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── metrics.py
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset should be placed in the `data/` folder:

```
data/
└── ds_v0.parquet
```

If the dataset is not included in the repository, download it from:

```
https://disk.yandex.ru/d/APXoRiQYP5_nyA
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

Run training from the `src/` directory:

```bash
cd src
python train.py
```

---

## Evaluation

After training, the script automatically evaluates the model on the validation and test sets.

Example output:

```
===== TEST METRICS =====
eval_loss: ...
eval_f1_macro: ...
eval_f1_clean: ...
...
```

---

## Results

(Add your results here)

| Class       | F1 score |
| ----------- | -------- |
| Clean       | 0.9438   |
| Num. flip   | 0.9460   |
| Entity Swap | 0.9820   |
| Neg. Flip   | 0.9949   |
| Macro       | 0.9667   |

---

## Dependencies

Tested with:

* torch 2.2.2
* transformers 4.38.2
* datasets 4.8.5
* scikit-learn 1.3.1
* numpy 1.26.4

---

## Notes

* The model is designed to be lightweight and efficient.
* Numeric features are explicitly modeled to improve detection of number-related hallucinations.
* The project was developed for educational during online NLP course from V. Malykh.
