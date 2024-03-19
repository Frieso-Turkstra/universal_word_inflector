import pandas as pd

if __name__ == "__main__":
    # Read in the wals data.
    df = pd.read_csv("data/wals.csv")

    # The data consists of 10 columns with general information, followed by 192
    # features, 12 of which are morphological.
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

    # Keep only the general information and morphological features.
    keep_columns = general_information + list(morphological_features.keys())
    df = df[keep_columns]

    # Only select languages that are available in both the shared task and WALS
    # and have annotations for all 12 morphological features
    wals_codes = ["aeg", "eng", "fin", "fre", "geo", "ger", "heb", "hun", "jpn", "rus", "spa", "swa", "tur"]
    df = df.loc[df["wals_code"].isin(wals_codes)]    

    # Save predictions
    df.to_json("features.jsonl", lines=True, orient="records")
