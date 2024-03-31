"""
This program explores all possible combinations of the morphological features
that are present in the WALS dataset. For each combination, the program checks
for each language in the shared task if it has annotations for each feature in 
the combination. Then, the user can specify certain constraints to find the
optimal feature combination, depending on the user's requirements. The user
can choose a feature combination to save to a file.
"""

import pandas as pd
import itertools
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file_path", "-o",
                        required=False,
                        help="Path to the output file.",
                        type=str,
                        default="features.jsonl",
                        )
    
    args = parser.parse_args()
    return args


def get_feature_combinations(dataframe, features, wals_codes):
    # Collect the available languages for every unique combination of features.
    # The key is a tuple of features, the value is a list with languages that 
    # have annotations for each of the features.
    results = dict()

    for length in range(1, len(features) + 1):
        for combination in itertools.combinations(features, length):
            # Get the row of the language and select the columns of the features in the current subset.
            # A language is stored if it does not have any NaNs for any of the features/columns.
            results[combination] = [
                wals_code for wals_code in wals_codes
                if not dataframe.loc[dataframe["wals_code"] == wals_code][list(combination)].isna().any().any()
                ]

    return results


def save(dataframe, valid_feature_combination, output_file_path):

    # Extract the features and languages.
    features, languages = valid_feature_combination

    # Keep the general information and relevant feature columns.
    general_information = [
        "wals_code", "iso_code", "glottocode", "Name", "latitude",
        "longitude", "genus", "family", "macroarea", "countrycodes"
    ]

    dataframe = dataframe[general_information + list(features)]

    # Select only the languages for which all features are annotated.
    dataframe = dataframe.loc[dataframe["wals_code"].isin(languages)]   

    # Save to file.
    dataframe.to_json(output_file_path, lines=True, orient="records")
    print("Successfuly saved!")


if __name__ == "__main__":

    # Read in the WALS data and command-line arguments.
    df = pd.read_csv("data/wals.csv")
    args = create_arg_parser()
    output_file_path = args.output_file_path

    # Only select languages that are available in both WALS and the shared task.
    # (24/26, excludes Ancient Greek and Sanskrit)
    wals_codes = [
        "amh", "aeg", "dsh", "eng", "fin", "fre", "heb", "hun", "arm", "ita", "jpn", "geo",
        "mcd", "rus", "spa", "swa", "tur", "nav", "arg", "alb", "ger", "sno", "blr", "khg"
        ]
    df = df.loc[df["wals_code"].isin(wals_codes)]

    # The data consists of 10 columns with general information, followed by 192
    # columns with features, 12 of which are morphological.
    general_information = [
        "wals_code", "iso_code", "glottocode", "Name", "latitude",
        "longitude", "genus", "family", "macroarea", "countrycodes"
    ]

    features = {
        "20A Fusion of Selected Inflectional Formatives": ["1 Exclusively concatenative", "2 Exclusively isolating", "3 Exclusively tonal", "4 Tonal/isolating", "5 Tonal/concatenative", "6 Ablaut/concatenative", "7 Isolating/concatenative"],
        "21A Exponence of Selected Inflectional Formatives": ["1 Monoexponential case", "2 Case + number", "3 Case + referentiality", "4 Case + TAM (tense-aspect-mood)", "5 No case"],
        "21B Exponence of Tense-Aspect-Mood Inflection": ["1 monoexponential TAM", "2 TAM+agreement", "3 TAM+agreement+diathesis", "4 TAM+agreement+construct", "5 TAM+polarity", "6 no TAM"],
        "22A Inflectional Synthesis of the Verb": ["1 0-1 category per word", "2 2-3 categories per word", "3 4-5 categories per word", "4 6-7 categories per word", "5 8-9 categories per word", "6 10-11 categories per word", "7 12-13 categories per word"],
        "23A Locus of Marking in the Clause": ["1 P is head-marked", "2 P is dependent-marked", "3 P is double-marked", "4 P has no marking", "5 Other types"],
        "24A Locus of Marking in Possessive Noun Phrases": ["1 Possessor is head-marked", "2 Possessor is dependent-marked", "3 Possessor is double-marked", "4 Possessor has no marking", "5 Other types"],
        "25A Locus of Marking: Whole-language Typology": ["1 Consistently head-marking", "2 Consistently dependent-marking", "3 Consistently double-marking", "4 Consistently zero-marking", "5 Inconsistent marking or other type"],
        "25B Zero Marking of A and P Arguments": ["1 Zero-marking", "2 Non-zero marking"],
        "26A Prefixing vs. Suffixing in Inflectional Morphology": ["1 Little or no inflectional morphology", "2 Predominantly suffixing", "3 Moderate preference for suffixing", "4 Approximately equal amounts of suffixing and prefixing", "5 Moderate preference for prefixing", "6 Predominantly prefixing"],
        "27A Reduplication": ["1 Productive full and partial reduplication", "2 Full reduplication only", "3 No productive reduplication"],
        "28A Case Syncretism": ["1 Inflectional case marking is absent or minimal", "2 Inflectional case marking is syncretic for core cases only", "3 Inflectional case marking is syncretic for core and non-core cases", "4 Inflectional case marking is never syncretic"],
        "29A Syncretism in Verbal Person/Number Marking": ["1 No subject person/number marking", "2 Subject person/number marking is syncretic", "3 Subject person/number marking is never syncretic"],
    }

    # Keep only the general information and morphological feature columns.
    keep_columns = general_information + list(features.keys())
    df = df[keep_columns]

    # Collect the available languages for every combination of features.
    print("Checking all feature combinations...")
    feature_combinations = get_feature_combinations(df, features, wals_codes) 

    # Try out different constraints to find the optimal combination.
    # If you want to include all the languages specified in `include_languages`,
    # you need to set `min_include_languages` equal to the length of `include_languages`.
    # Otherwise, you can use it for example to include at least 2 of the worst-
    # performing languages. Use ctrl-c to exit.
    while True:
        try:
            min_number_features = int(input("Minimum number of features: "))
            min_number_languages = int(input("Minimum number of languages: "))
            include_languages = set(input("Include languages: ").split())
            min_include_languages = int(input("Minimum number of included languages: "))
        except ValueError:
            print("Oops, you may have typed something wrong.")

        # Get combinations that meet a minimum number of features/languages
        # and contain at least `min_include_languages` of 'include_languages'.
        valid_feature_combinations = dict(filter(lambda x:
            len(x[0]) >= min_number_features and
            len(x[1]) >= min_number_languages and
            len(include_languages.intersection(x[1])) >= min_include_languages,
            feature_combinations.items()
            ))
        
        # Print out all the feature combinations that meet the constraints.
        for i, (features, languages) in enumerate(valid_feature_combinations.items()):
            print(f"FEATURE COMBINATION {i}")
            print("Features: ", features)
            print("Languages: ", languages)
        
        # Ask the user if they want to save the results.
        saving = input("Do you want to save the results? [y/n] ").lower()
        if saving.startswith("y"):

            # Ask which one needs to be saved unless there is only one.
            if len(valid_feature_combinations) == 1:
                idx = 0
            else:
                max_num = len(valid_feature_combinations)
                idx = int(input(f"Which one do you want to save? [0-{max_num-1}] "))
                if not (0 <= idx < max_num): 
                    print("Invalid reply.")
                    continue
            
            # Save results.
            features = list(valid_feature_combinations.keys())[idx]
            languages = valid_feature_combinations[features]
            save(df, (features, languages), output_file_path)
