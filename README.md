# Medical Abstract Classification with BiLSTM

A deep learning text classifier that reads a medical abstract and predicts which of five disease categories it belongs to. The project compares two ways of representing words: **FastText** embeddings trained on this corpus, and pretrained **BioWordVec** biomedical embeddings.

## The problem

Medical literature grows faster than anyone can read it. Automatically sorting abstracts by condition helps researchers and clinicians find relevant work. The task here is a five-class problem:

| Class | Training abstracts |
|---|---|
| Neoplasms | 2,530 |
| Digestive system diseases | 1,195 |
| Nervous system diseases | 1,540 |
| Cardiovascular diseases | 2,441 |
| General pathological conditions | 3,844 |

The classes are imbalanced, so results are reported with **macro-F1** as well as accuracy, and splits are stratified.

## Approach

1. **Preprocessing:** lowercasing, tokenising, lemmatising and stopword removal, while keeping negations ("no", "not") and domain punctuation such as hyphens, which carry meaning in medical text.
2. **Embeddings:**
   - **FastText:** 200-dimensional skip-gram vectors trained on the training abstracts. Subword n-grams give near-full coverage of rare and misspelled terms.
   - **BioWordVec:** 200-dimensional vectors pretrained on PubMed and MeSH. They cover about 78% of this corpus's vocabulary directly.
3. **Visualisation:** t-SNE plots of selected medical terms compare how each embedding space groups related concepts.
4. **Model:** a bidirectional LSTM (128 hidden units per direction) in PyTorch, initialised with each embedding set. The final forward and backward states feed a linear classifier.
5. **Training:** Adam optimiser, 90/10 stratified train/validation split, and early stopping on validation accuracy.
6. **Evaluation:** accuracy, macro-F1 and a per-class report on a held-out test set of 2,888 abstracts.
7. **Interactive inference:** a small widget that loads the saved weights, lets you pick an abstract from the test set and shows the true and predicted label, with no retraining.

## Results

On the held-out test set:

| Embedding | Accuracy | Macro-F1 |
|---|---|---|
| FastText (trained on corpus) | 0.596 | 0.585 |
| BioWordVec (pretrained biomedical) | 0.601 | 0.580 |

Per-class F1:

| Class | FastText | BioWordVec |
|---|---|---|
| Neoplasms | 0.748 | 0.744 |
| Digestive system diseases | 0.454 | 0.469 |
| Nervous system diseases | **0.575** | 0.478 |
| Cardiovascular diseases | 0.698 | **0.729** |
| General pathological conditions | 0.451 | 0.480 |

**What this shows**
- The two embeddings perform almost the same overall, but they are good at different things. BioWordVec is stronger on cardiovascular, digestive and general conditions. FastText is clearly better on nervous system diseases, mainly through higher recall.
- BioWordVec reaches its best validation score after one epoch, so pretrained biomedical knowledge helps the model learn quickly. Both models start to overfit after one or two epochs, which early stopping handles.
- "General pathological conditions" is the hardest class. It is the largest and most varied, and it overlaps with every other category.

## Repository structure

```
medical_abstract_classifier.ipynb   full pipeline: preprocessing, embeddings, t-SNE, BiLSTM, evaluation, inference UI
requirements.txt                    Python packages
data/                               not included (see below)
models/                             created when you run the notebook
```

## How to run

1. Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Create a `data/` folder next to the notebook containing:
   - `medical_tc_train.csv`, `medical_tc_test.csv` and `medical_tc_labels.csv`
   - `bio_embedding_extrinsic`, the BioWordVec vectors (about 1.9 GB, word2vec binary format), available from the BioWordVec project
3. Open the notebook and run all cells. Trained weights and metadata are saved to `models/`, and section 6 loads them for interactive predictions.

The notebook also runs in Google Colab: mount your Drive and set `DATA_DIR` to the folder holding the data.

## Data

The dataset and the BioWordVec file are not included in this repository because of their size and redistribution terms. Each record is a medical abstract with an integer condition label (1 to 5).

## Tools

Python, pandas, NLTK, gensim (FastText, KeyedVectors), PyTorch, scikit-learn, matplotlib, ipywidgets

## Author

Muneef Muhammed, Master of Data Science, University of Western Australia
[Portfolio](https://muneef97.github.io) · [LinkedIn](https://www.linkedin.com/in/muneef-muhammed/)
