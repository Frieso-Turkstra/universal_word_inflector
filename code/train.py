from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
import numpy as np
from langchain.evaluation import load_evaluator, ExactMatchStringEvaluator
import os
import torch
import logging


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", "-l", required=False, nargs="+", help="List of Wals-codes to be trained on.", type=str, default="amh arz dan eng fin fra grc heb hun hye ita jap kat mkd rus spa swa tur nav afb sqi deu sme bel klr san")
    parser.add_argument("--model_size", "-ms", required=False, help="Size of the byT5 model.", type=str, default="base", choices=["small", "base", "large", "xl", "xxl"])
    parser.add_argument("--remove_features", "-rf", required=False, help="Train without the features.", action="store_true")
    parser.add_argument("--parameter_efficient_fine_tuning", "-peft", required=False, help="Use parameter efficient fine-tuning.", action="store_true")
    parser.add_argument("--output_file_path", "-o", required=False, help="Path to output file.", type=str, default="predictions.jsonl")
    
    args = parser.parse_args()
    return args


def preprocess(file_path, language, remove_features):
    # Read in the data, lemma and features are combined into one input column.
    df = pd.read_table(file_path, names=["lemma", "features", "target"])

    df["input"] = task_prefix + df["lemma"]
    if not remove_features:
        df["input"] += " " + df["features"]
    
    # Eos.token is added to target so the model learns when to stop generation.
    df["target"] = df["target"] + tokenizer.eos_token

    # Add language column so we can shuffle and still evaluate per language.
    df["language"] = language

    return df


def tokenize(examples, tokenizer, max_source_length, max_target_length):
    # Encode the inputs.
    encoding = tokenizer(
        examples["input"],
        padding="max_length",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    
    # Encode the targets.
    target_encoding = tokenizer(
        examples["target"],
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids

    # Replace padding token id's of the labels by -100 so it's ignored by the loss.
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute exact match accuracy.
    evaluator = ExactMatchStringEvaluator()

    avg_accuracy = np.mean(list(map(
        lambda x: evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
        zip(decoded_preds, decoded_labels)
        )))

    # Compute Levenshtein distance.
    levenshtein_evaluator = load_evaluator(
        "string_distance",
        distance='levenshtein'
    )
    
    avg_levensthein_distance = np.mean(list(map(
        lambda x: levenshtein_evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
        zip(decoded_preds, decoded_labels)
    )))
    
    return {"accuracy": avg_accuracy, "levenshtein": avg_levensthein_distance}


def fine_tune(model_name, tokenizer, tokenized_datasets, peft):

    # Load the model.
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Use parameter-efficient fine-tuning if enabled.
    if peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    train_args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        #eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        #save_steps=200,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        optim="adafactor",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets["trn"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # Save best model.
    best_model_path = model_name + "/best/"
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.save_model(best_model_path)


def test(model_path, test_tokenized_dataset, max_length, peft):
    # Load the best model.
    if peft:
        config = PeftConfig.from_pretrained(model_path)
        base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Prepare dataloader.
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

    # Generate text for each batch.
    with torch.no_grad():
        all_predictions = [model.generate(**batch, max_length=max_length) for batch in dataloader]

    # Flatten predictions.
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    # Turn labels to np.array so -100 get correctly converted to pad tokens.
    all_labels = np.array(test_tokenized_dataset["labels"])    

    # Compute metrics.
    predictions_labels = [all_predictions_flattened, all_labels]
    results = compute_metrics(predictions_labels)

    # Return the results and decoded predictions.
    decoded_predictions = tokenizer.batch_decode(all_predictions_flattened, skip_special_tokens=True)

    return results, decoded_predictions


if __name__ == "__main__":

    # Extract command line arguments.
    args = create_arg_parser()
    model_name = "google/byt5-" + args.model_size
    remove_features = args.remove_features
    languages = args.languages 
    peft = args.parameter_efficient_fine_tuning
    output_file_path = args.output_file_path
    
    # Set up logging.
    logging.basicConfig(filename="logs.log", encoding="utf-8", level=logging.DEBUG)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some hyperparameters
    # task prefix is not important in single-task finetuning unless it is related
    # to a task seen in training. (see sigmorphon.py/tasks.py in byt5 github)
    task_prefix = ""
    max_source_length = 128
    max_target_length = 64
    max_length = 64 # alternatively, max_new_tokens

    # Create tokenized dataset for each split.
    print("Loading data...")
    datasets = DatasetDict()

    for split in ("trn", "dev", "tst"):
        df = pd.concat([preprocess(f"data/{lang}.{split}", lang, remove_features) for lang in languages], ignore_index=True)
        datasets[split] = Dataset.from_pandas(df)

    tokenized_datasets = datasets.map(
        lambda x: tokenize(x, tokenizer, max_source_length, max_target_length),
        batched=True
        )
    
    # Shuffle the languages so the model does not learn them in order.
    tokenized_datasets["trn"] = tokenized_datasets["trn"].shuffle()
    
    # Only keep columns needed for training: input_ids, attention_mask, labels.
    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].remove_columns(["lemma", "features", "target", "input", "language"])

    # Finetune and test the model.
    print("Fine tuning...")
    fine_tune(model_name, tokenizer, tokenized_datasets, peft)

    print("Predicting...")
    results, predictions = test(f"{model_name}/best", tokenized_datasets["tst"], max_length, peft)

    # Save results/predictions.
    predictions_df = pd.DataFrame({"label": predictions, "language": datasets["tst"]["language"]})
    predictions_df.to_json(output_file_path, lines=True, orient="records")
    print("Successfully saved predictions!")
    print(results)
    logging.info(results)
