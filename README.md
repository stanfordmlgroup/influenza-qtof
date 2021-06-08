# Influenza Diagnosis using Q-TOF LC/MS

Novel metabolomics approach combined with machine learning for the diagnosis of influenza from nasopharyngeal specimens

# Usage

## Restrospective Dataset

To train models with cross validation and test on a held-out set of samples, run `train_and_test.py`. Usage: `./train_and_test.py -o <output_dir> -d <data_csv> -l <labels_csv>`.

## Prospective Dataset
To validate models trained with retrospective data on an unseen prospective dataset, run `prospective_train.py` to produce the necessary model checkpoints, followed by `prospective_evaluate.py` to evaluate the models on the prospective set. Usage:
```
python prospective_train.py
python prospective_evaluate.py
```