# universal_word_inflector

## Fine-tune byT5 

### Overview
This program fine-tunes a byT5 model on the task of morphological inflection. The fine-tuned model is saved and evaluated on exact match accuracy and Levenshtein distance.  

### Command

```sh
python code/train.py --languages spa tur swa --model_size small 
```

- **--languages**: [required] Wals-codes of languages on which the model will be trained. 
- **--model_size**: [optional] Size of the byT5 model, options are small, base, large, xl, xxl. 
