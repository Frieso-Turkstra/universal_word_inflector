# universal_word_inflector

## Data and Dependencies

The training data can be downloaded from: https://github.com/sigmorphon/2023InflectionST/tree/main/part1/data

Run the following command to install the necessary dependencies:

```sh
pip install -r requirements.txt
```

## Fine-tuning and inference with ByT5 

### Overview
This program fine-tunes a ByT5 model on the task of morphological inflection. The fine-tuned model is saved and evaluated on exact match accuracy and Levenshtein distance. The program logs the results and outputs a file with the predictions. It is also possible to run the program in inference-only mode, which skips the fine-tuning and can be used to predict unseen data with an already fine-tuned model.

### Command

```sh
python code/train.py --iso_codes spa tur swa --model google/byt5-small --output_file_path model_predictions_small.jsonl
```

- **--iso_codes**: [optional] Iso-codes of languages on which the model will be trained, defaults to all languages of the shared task. 
- **--model**: [optional] T5 model used for fine-tuning and/or inference, defaults to "google/byt5-base".
-**--inference_only**: [optional] Skip the fine-tuning phase. 
- **--remove_features**: [optional] Train with only the lemma as input.
- **--output_file_path**: [optional] Path to which the predictions will be saved, defaults to "model_predictions.jsonl".

## Evaluate per Language

### Overview
This program calculates the exact match accuracy and Levenshtein distance per language. It takes as input a file with predictions, automatically infers which labels it needs and outputs a file with the evaluation metrics per language.

### Command

```sh
python code/eval.py --input_file_path model_predictions_small.jsonl --output_file_path scores_small.jsonl
```

- **--input_file_path**: [required] Path to the file with the predictions.
- **--output_file_path**: [optional] Path to which the scores will be saved, defaults to "scores.jsonl".

## Select and Extract Features

### Overview
This program explores all possible combinations of the morphological features that are present in the WALS dataset. For each combination, the program checks for each language in the shared task if it has annotations for each feature in the combination. Then, the user can specify certain constraints to find the optimal feature combination, depending on the user's requirements. The user can choose a feature combination to save to a file.

### Command

```sh
python code/features.py --output_file_path features.jsonl
```

- **--output_file_path**: [optional] Path to which the features will be saved, defaults to "features.jsonl".

## Analyze Results

### Overview
This program tries to predict the accuracy scores of a language on the task of morphological (re-)inflection based on its morphological features. In total, three models generate predictions:
1) Linear regression model with train-test split
2) Linear regression model with cross-validation
3) Multi-layer perceptron

All models are evaluated on Mean Absolute Error and Mean Squared Error. Additionally, visualisations can be generated.

### Command

```sh
python code/regression.py --accuracy_file_path scores_small.jsonl --features_file_path features.jsonl --plot
```

- **--accuracy_file_path**: [required] Path to the file with the accuracy scores.
- **--features_file_path**: [required] Path to the file with the features.
- **--plot**: [optional] Plot the cross-validated predictions vs. actual values.

## Statistical Analysis (stats folder)
The code for statistical analysis has been written in R. To run it, open the file NLP_UNINFL_2A_StatsScript.R in RStudio. 

**Data:** Make sure that the files fulldata.csv and scores_permodel.csv are in the same folder as the R script.

**Packages:** pacman::p_load on the first line takes care of installing and updating the necessary packages.
