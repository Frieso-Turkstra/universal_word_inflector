"""
This program calculates the exact match accuracy and Levenshtein distance per language.
It takes as input a file with predictions, automatically infers which labels it
needs and outputs a file with the evaluation metrics per language.
"""

from langchain.evaluation import ExactMatchStringEvaluator
from langchain.evaluation import load_evaluator
import pandas as pd
import numpy as np
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_path", "-i",
                        required=True,
                        help="Path to the prediction file.",
                        type=str,
                        )
    parser.add_argument("--output_file_path", "-o",
                        required=False,
                        help="Path to the output file.",
                        type=str,
                        default="scores.jsonl",
                        )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__": 

    # Extract command line arguments.
    args = create_arg_parser()
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    # Read in the data.
    df = pd.read_json(input_file_path, lines=True)
    iso_codes = df["iso_code"].unique()
    df.set_index("iso_code", inplace=True)

    # Get the evaluation metrics per language.
    scores = []
    for iso_code in iso_codes:

        # Get the predictions and labels for the language with `iso_code`.
        predictions = df.loc[iso_code]["label"]
        
        label_df = pd.read_table(f"data/{iso_code}.tst", names=["lemma", "features", "target"])
        labels = label_df["target"]

        # Compute the average exact match accuracy.
        evaluator = ExactMatchStringEvaluator()

        avg_accuracy = np.mean(list(map(
            lambda x: evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
            zip(predictions, labels)
            )))
        
        # Compute the average Levenshtein distance.
        levenshtein_evaluator = load_evaluator(
            "string_distance",
            distance='levenshtein'
        )
        
        avg_levensthein_distance = np.mean(list(map(
            lambda x: levenshtein_evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
            zip(predictions, labels)
        )))

        # Save scores with the iso_code as a triple.
        scores.append((iso_code, avg_accuracy, avg_levensthein_distance))

    # Save the results.
    df = pd.DataFrame(scores, columns=["iso_code", "accuracy", "levenshtein"])
    df.to_json(output_file_path, lines=True, orient="records")
    
    # Print the results with the full names and sorted from high to low.
    iso2name = {
        "amh": "Amharic", "arz": "Arabic (Egyptian)", "dan": "Danish",
        "eng": "English", "fin": "Finnish", "fra": "French",
        "grc": "Ancient Greek", "heb": "Hebrew", "hun": "Hungarian",
        "hye": "Armenian", "ita": "Italian", "jap": "Japanese",
        "kat": "Georgian", "mkd": "Macedonian", "rus": "Russian",
        "spa": "Spanish", "swa": "Swahili", "tur": "Turkish", "nav": "Navajo",
        "afb": "Arabic (Gulf)", "sqi": "Albanian", "deu": "German",
        "sme": "Sami", "bel": "Belarusian", "klr": "Khaling", "san": "Sanskrit"
        }
    df.insert(1, "language", df["iso_code"].apply(lambda x: iso2name[x]))

    print(df.sort_values("accuracy", ascending=False, ignore_index=True))
    print("Average accuracy: ", df["accuracy"].mean())
    print("Average Levenshtein distance: ", df["levenshtein"].mean())

    