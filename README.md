# Decision Trees using Python — Assignment 3 (Learn It, Then Prove It)

B.Tech Semester VII · Artificial Intelligence (20CST401) · Module V, Assignment 3
Infosys Springboard course **"Decision Trees using Python"**, for roll numbers 51–75 (this submission: **roll no. 70**)

## Assignment

Part B of the assignment asks for evidence from my own code and output for each of the course's nine learning outcomes. The notebook `Decision_Trees_Assignment.ipynb` provides it, using the UCI Bank Marketing dataset in `data/`:

| # | Learning outcome | Notebook section |
|---|---|---|
| 1 | Induct a decision tree to perform classification | 7 |
| 2 | Explain the attribute selection measures for classification (entropy, information gain, Gini) | 8 |
| 3 | Predict the outcome variable with the tree | 9 |
| 4 | Evaluate prediction accuracy (accuracy, precision, recall, F1, confusion matrix) | 10 |
| 5 | Induct a decision tree to perform regression | 11 |
| 6 | Explain the attribute selection measure for regression (MSE / variance reduction) | 12 |
| 7 | Evaluate the regression tree with R² and RMSE | 13 |
| 8 | Induct model trees that fit piece-wise models after splitting | 14 |
| 9 | Prune a tree with stopping and pruning techniques (max_depth, min_samples_split, min_samples_leaf, ccp_alpha) | 15 |

The mapping below, Section 17 of the notebook, and `Outcome_Mapping.md` each link every outcome to the exact output that proves it. Section 19 has 35 viva questions with answers.

Part A (Springboard completion screenshot and certificate) has to be added separately by the student.

## Mapping: nine learning outcomes → evidence

Notebook: `Decision_Trees_Assignment.ipynb` · Data: `data/bank-additional.csv` (4,119 clients, UCI Bank Marketing) · 80/20 split, `random_state=42`

| Outcome | Evidence from my project (actual outputs) | Screenshot |
|---|---|---|
| 1. Classification tree induction | DecisionTreeClassifier (criterion=gini) trained on 3295 clients; depth 25, 436 leaves; train acc 1.0000, test acc 0.8265; root split nr.employed <= 5087.65; tree diagram | [outcome_01_classification.png](images/outcome_01_classification.png) |
| 2. Attribute selection (classification) | Root Gini 0.1951, entropy 0.4986 bits; search of all 62 features: best Gini decrease 0.0320, best information gain 0.0831 (nr.employed), same root as sklearn for gini and entropy; gini/entropy/log_loss compared | [outcome_02_attribute_selection.png](images/outcome_02_attribute_selection.png) |
| 3. Prediction | predict()/predict_proba() on 8 unseen test clients: 7 correct, 1 wrong; decision path of one client traced node by node | [outcome_03_prediction.png](images/outcome_03_prediction.png) |
| 4. Classification evaluation | Accuracy 0.8265, precision 0.2022, recall 0.2000, F1 0.2011; confusion matrix TN 663 / FP 71 / FN 72 / TP 18; always-'no' baseline accuracy 0.8908 with recall 0 | [outcome_04_accuracy.png](images/outcome_04_accuracy.png) |
| 5. Regression tree induction | DecisionTreeRegressor predicting age from client profile; max_depth=5 chosen by 5-fold CV; 31 leaves; prediction table and tree diagram | [outcome_05_regression.png](images/outcome_05_regression.png) |
| 6. Attribute selection (regression) | criterion=squared_error; root MSE = variance = 106.52; best split marital_single <= 0.5 (child MSE 87.53), same as sklearn root; 4 criteria compared | [outcome_06_regression_selection.png](images/outcome_06_regression_selection.png) |
| 7. R-squared and RMSE | Test R2 = 0.3227, RMSE = 8.46 years (mean-only baseline RMSE 10.27); both verified from their formulas; fully grown tree test R2 0.1920 | [outcome_07_r2_rmse.png](images/outcome_07_r2_rmse.png) |
| 8. Model tree (piece-wise models) | Split euribor3m <= 2.57 + one LinearRegression per region; test R2 0.931 vs 0.885 (global line) vs 0.834 (constant leaves); age task 0.345 vs 0.158 | [outcome_08_piecewise.png](images/outcome_08_piecewise.png) |
| 9. Pruning | Before: depth 25, 436 leaves, test acc 0.8265. After ccp_alpha=0.00175: depth 2, 3 leaves, test acc 0.9017; max_depth / min_samples_split / min_samples_leaf (CV-tuned) also compared | [outcome_09_pruning.png](images/outcome_09_pruning.png) |

All values come from the models trained in Sections 7–15 of the notebook. Classification uses Model B (without the leaky `duration` column); regression predicts `age`.

## Dataset

**Bank Marketing (with social/economic context)**, from Moro, Cortez & Rita (2014), *Decision Support Systems*, UCI ML Repository. It comes from a Portuguese bank's phone campaigns.

| File in `data/` | Use |
|---|---|
| `bank-additional.csv` | **Main dataset.** 4,119 rows × 21 columns, semicolon-separated. It is an official 10% random sample of the full file |
| `bank-additional-full.csv` | 41,188 rows. Loaded only to confirm it has the same columns and class balance |
| `bank-additional-names.txt` | Official data dictionary |
| `.DS_Store`, `.Rhistory` | Archive leftovers (macOS metadata, R history). Not used |

- **Classification target:** `y`, meaning whether the client subscribed to a term deposit (yes/no). It is imbalanced: 10.95% "yes".
- **`duration` (call length) is excluded from the main model.** It is only known after the call, so using it is target leakage. Section 10.4 compares Model A (with `duration`) against Model B (without it).
- **Regression target:** `age`, predicted from the client's profile (`job`, `marital`, `education`, `default`, `housing`, `loan`). This is a genuine column of the dataset and nothing is fabricated. `y` is never used as a regression target. `duration` was rejected with evidence (R² ≈ 0 when predicted from pre-call features).
- **Piece-wise demo:** a model tree (tree split followed by a linear regression per region), shown on two real columns (`euribor3m` → `nr.employed`) and on the age task.
- There are no NaNs and no duplicate rows. The label `"unknown"` in categorical columns is kept as its own category.

The files in `data/` are only read and never modified.

## How to run

Requirements: Python 3.9 or newer (tested with Python 3.14, pandas 3.0, scikit-learn 1.8).

```bash
pip install -r requirements.txt
```

To open the notebook interactively:

```bash
python -m notebook Decision_Trees_Assignment.ipynb
```

Then choose **Kernel → Restart & Run All**. The notebook is saved with all outputs already executed, so it can also be read without running it.

To re-execute every cell from the command line (takes about 45 seconds):

```bash
python -m nbconvert --to notebook --execute --inplace Decision_Trees_Assignment.ipynb
```

Run the notebook from this folder, because it reads `data/bank-additional.csv` through a relative path. `random_state=42` is used everywhere, so the results can be reproduced exactly.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (1.2 or newer, for `OneHotEncoder(sparse_output=...)` and `criterion="log_loss"`), and `notebook`/`nbconvert` to run it. See `requirements.txt`.

## Files and folders

```
.
├── Decision_Trees_Assignment.ipynb   # main notebook (executed, with outputs)
├── Outcome_Mapping.md                # one-page written mapping: 9 outcomes -> evidence
├── images/                           # outcome_01 ... outcome_09 evidence images (one per learning outcome)
├── README.md                         # this file
├── requirements.txt                  # Python dependencies
└── data/                             # original dataset (unchanged)
    ├── bank-additional.csv
    ├── bank-additional-full.csv
    └── bank-additional-names.txt
```

## Key results (test set, from the executed notebook)

| Model | Depth | Leaves | Test accuracy | Test F1 ("yes") |
|---|---|---|---|---|
| Always "no" (baseline) | – | – | 0.891 | 0.000 |
| Unpruned Gini tree (Model B) | 25 | 436 | 0.827 | 0.201 |
| Cost-complexity pruned tree (Model B) | 2 | 3 | 0.902 | 0.243 |
| Depth-5 tree **with** `duration` (Model A, leakage benchmark) | 5 | 32 | 0.909 | 0.510 |

| Regression model (target: `age`) | R² | RMSE (years) |
|---|---|---|
| Always predict the mean | −0.001 | 10.28 |
| Regression tree, fully grown | 0.192 | 9.24 |
| Regression tree, max_depth=5 (chosen by cross-validation) | 0.323 | 8.46 |
| Model tree (depth 1, a linear model in each leaf) | 0.345 | 8.31 |
