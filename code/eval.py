import pandas as pd
from langchain.evaluation import ExactMatchStringEvaluator
import numpy as np
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file_path", "-i", required=True, help="Path to prediction file", type=str)
    parser.add_argument("--output_file_path", "-o", required=False, help="Path to output file", type=str, default="scores.jsonl")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    args = create_arg_parser()

    # Read in the data
    df = pd.read_json(args.input_file_path, lines=True)
    languages = df["language"].unique()
    df.set_index("language", inplace=True)

    # Get scores per language
    scores = dict()
    for language in languages:

        # Get the predictions and labels for `language`.
        predictions = df.loc[language]["label"]
        
        lang_df = pd.read_table(f"data/{language}.tst", names=["lemma", "features", "target"])
        labels = lang_df["target"]

        # Compute the average exact match accuracy.
        evaluator = ExactMatchStringEvaluator()

        avg_accuracy = np.mean(list(map(
            lambda x: evaluator.evaluate_strings(prediction=x[0], reference=x[1])["score"],
            zip(predictions, labels)
            )))

        # Save score to dictionary.
        scores[language] = avg_accuracy

    # Save and print the results.
    df = pd.DataFrame(scores.items(), columns=["iso_code", "accuracy"])
    df.to_json(args.output_file_path, lines=True, orient="records")
    
    print(df.sort_values("accuracy", ascending=False))
    print("Average accuracy: ", df["accuracy"].mean())

    
