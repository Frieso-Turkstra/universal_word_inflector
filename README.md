# universal_word_inflector

## Fine-tune byT5 

### Overview
This program fine-tunes a byT5 model on the task of morphological inflection. The fine-tuned model is saved and evaluated on exact match accuracy and Levenshtein distance.  

### Command

```sh
python code/train.py --languages spa tur swa --model_size small -peft --output_file_path predictions_small.jsonl
```

- **--languages**: [optional] Wals-codes of languages on which the model will be trained, defaults to all languages of the shared task. 
- **--model_size**: [optional] Size of the byT5 model, options are small, base, large, xl, xxl, defaults to base.
- **--parameter_efficient_fine_tuning**: [optional] Use parameter-efficient fine-tuning.
- **--remove_features**: [optional] Train with only the lemma as input.
- **--output_file_path**: [optional] Path to which the predictions will be saved, defaults to "predictions.jsonl".
