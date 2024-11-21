from transformers import pipeline
import pandas as pd
raw_data = pd.read_excel("initial_data.xlsx")
data = raw_data[['Tweets']]

## Defining labels for annotation
labels = ["Positive", "Negative", "Neutral"]

## Model 1: Using BART model (Pre- trained Zero- Shot Classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

## Function to annotate the statement
def annotate_statement_BART(statement):
    try:
        result = classifier(statement, candidate_labels=labels)

        ## Printing labels with scores
        # print(result)

        ## Label with highest Score
        return result['labels'][0]

    except Exception as e:
        # print(e)
        return "Error"


data['Annotation'] = data['Tweets'].apply(annotate_statement_BART)
# print(data)
## Saving results to an excel file.
output_file = "annotated_statements.xlsx"
data.to_excel(output_file, index=False)

