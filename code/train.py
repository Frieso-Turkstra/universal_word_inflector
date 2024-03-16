from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import pandas as pd
import argparse
import numpy as np
from langchain.evaluation import load_evaluator, ExactMatchStringEvaluator
import os
import torch
import json
import logging

"""
TODO
- figure out dynamic padding, @longest in data collator?
"""

def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test.", type=str)
    parser.add_argument("--languages", "-l", required=True, help="List of Wals codes to be learned.", type=json.loads)

    args = parser.parse_args()
    return args


def preprocess(file_path):
    # Read in the data, lemma and features are combined into one input column.
    # Eos.token is added to target so the model learns when to stop generation.
    df = pd.read_table(file_path, names=["lemma", "features", "target"])
    df["input"] = task_prefix + df["lemma"] + " " + df["features"]
    df["target"] = df["target"] + tokenizer.eos_token
    return df


def tokenize(examples, tokenizer, max_source_length, max_target_length):
    # Encode the inputs
    encoding = tokenizer(
        examples["input"],
        padding="max_length",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    
    # Encode the targets
    target_encoding = tokenizer(
        examples["target"],
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids

    # Replace padding token id's of the labels by -100 so it's ignored by the loss
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
    
    # Compute exact match accuracy
    evaluator = ExactMatchStringEvaluator()

    avg_accuracy = np.mean(list(map(
        lambda x: evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
        zip(decoded_preds, decoded_labels)
        )))

    # Compute Levenshtein distance
    levenshtein_evaluator = load_evaluator(
        "string_distance",
        distance='levenshtein'
    )
    
    avg_levensthein_distance = np.mean(list(map(
        lambda x: levenshtein_evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
        zip(decoded_preds, decoded_labels)
    )))
    
    return {"accuracy": avg_accuracy, "levenshtein": avg_levensthein_distance}


def fine_tune(model_name, tokenizer, tokenized_datasets):

    # Load the model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
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
        #report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets["trn"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save best model.
    best_model_path = "finetuned/" + model_name + "/best/"
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.save_model(best_model_path)


def test(model_path, test_tokenized_dataset, max_length):
    # Load the best model
    best_model = T5ForConditionalGeneration.from_pretrained(model_path)

    # prepare dataloader
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

    # generate text for each batch
    all_predictions = [best_model.generate(**batch, max_length=max_length) for batch in dataloader]

    # flatten predictions
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    # turn labels to np.array so -100 get correctly converted to pad tokens
    all_labels = np.array(test_tokenized_dataset["labels"])    

    # compute metrics
    predictions_labels = [all_predictions_flattened, all_labels]
    results = compute_metrics(predictions_labels)

    return results, all_predictions_flattened


if __name__ == "__main__":

    # Extract command line arguments
    args = create_arg_parser()
    model_name = args.model
    language = args.languages.pop()

    # Set up logging
    logging.basicConfig(filename="logs.log", encoding="utf-8", level=logging.DEBUG)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some hyperparameters
    # task prefix is not important in single-task finetuning unless it is related
    # to a task seen in training
    task_prefix = ""
    max_source_length = 512
    max_target_length = 128
    max_length = 100 # alternatively, max_new_tokens

    # Create tokenized dataset for each split
    datasets = DatasetDict()

    for split in ("trn", "dev", "tst"):
        df = preprocess(f"data/{language}.{split}")
        datasets[split] = Dataset.from_pandas(df)

    tokenized_datasets = datasets.map(
        lambda x: tokenize(x, tokenizer, max_source_length, max_target_length),
        batched=True
        )
    
    # Only keep columns needed for training: input_ids, attention_mask, labels
    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].remove_columns(["lemma", "features", "target", "input"])

    # Finetune and test the model
    fine_tune(model_name, tokenizer, tokenized_datasets)
    results, predictions = test(f"finetuned/{model_name}/best", tokenized_datasets["tst"], max_length)

    # Save results/predictions.
    predictions_file_path = f"predictions.jsonl"

    print(results)
    logging.info(results)
    predictions_df = pd.DataFrame({"label": predictions})
    predictions_df.to_json(predictions_file_path, lines=True, orient="records")
    print("Successfully saved predictions!")
