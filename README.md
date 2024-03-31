# universal_word_inflector

## Fine-tune ByT5 

### Overview
This program fine-tunes a ByT5 model on the task of morphological inflection. The fine-tuned model is saved and evaluated on exact match accuracy and Levenshtein distance. The program logs the results and outputs a file with the predictions.

### Command

```sh
python code/train.py --iso_codes spa tur swa --model_size small --output_file_path model_predictions_small.jsonl
```

- **--iso_codes**: [optional] Iso-codes of languages on which the model will be trained, defaults to all languages of the shared task. 
- **--model_size**: [optional] Size of the byT5 model, options are small, base, large, xl, xxl, defaults to base.
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
1) a linear regression model
2) a linear regression model with cross-validation
3) a multi-layer perceptron
All models are evaluated on Mean Absolute Error and Mean Squared Error. Additionally, visualisations can be generated.

### Command

```sh
python code/regression.py --accuracy_file_path scores_small.jsonl --features_file_path features.jsonl --plot
```

- **--accuracy_file_path**: [required] Path to the file with the accuracy scores.
- **--features_file_path**: [required] Path to the file with the features.
- **--plot**: [optional] Plot the cross-validated predictions vs. actual values.
