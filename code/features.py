import pandas as pd
import itertools


def get_feature_combinations(features, wals_codes):
    # Collect the available languages for every unique combination of features
    # The key is a tuple of features, the value is a list with languages that 
    # have annotation for each of the features.
    results = dict()

    for length in range(1, len(features) + 1):
        for combination in itertools.combinations(features, length):
            # Get the row of the language and select the columns of the features in the current subset
            # a language is stored if it does not have any NaNs for any of the features/columns
            results[combination] = [
                wals_code for wals_code in wals_codes
                if not df.loc[df["wals_code"] == wals_code][list(combination)].isna().any().any()
                ]

    return results


if __name__ == "__main__":

    # Read in the WALS data.
    df = pd.read_csv("data/wals.csv")

    # The data consists of 10 columns with general information, followed by 192
    # columns with features, 12 of which are morphological.
    general_information = [
        "wals_code", "iso_code", "glottocode", "Name", "latitude",
        "longitude", "genus", "family", "macroarea", "countrycodes"
    ]

    morphological_features = {
        "20A Fusion of Selected Inflectional Formatives": ["Exclusively concatenative", "Exclusively isolating", "Exclusively tonal", "Tonal/isolating", "Tonal/concatenative", "Ablaut/concatenative", "Isolating/concatenative"],
        "21A Exponence of Selected Inflectional Formatives": ["Monoexponential case", "Case + number", "Case + referentiality", "Case + TAM (tense-aspect-mood)", "No case"],
        "21B Exponence of Tense-Aspect-Mood Inflection": ["monoexponential TAM", "TAM+agreement", "TAM+agreement+diathesis", "TAM+agreement+construct", "TAM+polarity", "no TAM"],
        "22A Inflectional Synthesis of the Verb": ["0-1 category per word", "2-3 categories per word", "4-5 categories per word", "6-7 categories per word", "8-9 categories per word", "10-11 categories per word", "12-13 categories per word"],
        "23A Locus of Marking in the Clause": ["P is head-marked", "P is dependent-marked", "P is double-marked", "P has no marking", "Other types"],
        "24A Locus of Marking in Possessive Noun Phrases": ["Possessor is head-marked", "Possessor is dependent-marked", "Possessor is double-marked", "Possessor has no marking", "Other types"],
        "25A Locus of Marking: Whole-language Typology": ["Consistently head-marking", "Consistently dependent-marking", "Consistently double-marking", "Consistently zero-marking", "Inconsistent marking or other type"],
        "25B Zero Marking of A and P Arguments": ["Zero-marking", "Non-zero marking"],
        "26A Prefixing vs. Suffixing in Inflectional Morphology": ["Little or no inflectional morphology", "Predominantly suffixing", "Moderate preference for suffixing", "Approximately equal amounts of suffixing and prefixing", "Moderate preference for prefixing", "Predominantly prefixing"],
        "27A Reduplication": ["Productive full and partial reduplication", "Full reduplication only", "No productive reduplication"],
        "28A Case Syncretism": ["Inflectional case marking is absent or minimal", "Inflectional case marking is syncretic for core cases only", "Inflectional case marking is syncretic for core and non-core cases", "Inflectional case marking is never syncretic"],
        "29A Syncretism in Verbal Person/Number Marking": ["No subject person/number marking", "Subject person/number marking is syncretic", "Subject person/number marking is never syncretic"],
    }

    # Keep only the general information and morphological feature columns.
    keep_columns = general_information + list(morphological_features.keys())
    df = df[keep_columns]

    # Only select languages that are available in both WALS and the shared task (23)
    wals_codes = ["alb", "amh", "aeg", "arg", "arm", "dsh", "eng", "fin", "fre", "geo", "ger", "heb", "hun", "ita", "jpn", "khg", "mcd", "nav", "blr", "rus", "spa", "swa", "tur"]
    df = df.loc[df["wals_code"].isin(wals_codes)]

    # If you want to explore the optimal combination of number of features and
    # languages with annotations for those features, set explore to True.
    explore = False
    min_number_features = 12
    min_number_languages = 1

    if explore:
        # Collect all the annotated languages for every combination of features.
        feature_combinations = get_feature_combinations(morphological_features, wals_codes) 

        # Get combinations that meet a minimum number of features/languages
        valid_feature_combinations = dict(filter(
            lambda x: len(x[0]) >= min_number_features and len(x[1]) >= min_number_languages,
            feature_combinations.items()
            ))
        
        print(valid_feature_combinations)
            
    # With a minimum number of features of 12, there are 13 languages available.
    # Remove the other languages.
    selected_wals_codes = ["aeg", "eng", "fin", "fre", "geo", "ger", "heb", "hun", "jpn", "rus", "spa", "swa", "tur"]
    df = df.loc[df["wals_code"].isin(selected_wals_codes)]    

    # Save to file.
    df.to_json("features.jsonl", lines=True, orient="records")
