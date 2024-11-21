from transformers import pipeline
import pandas as pd


raw_data = pd.read_excel("initial_data.xlsx")
data = raw_data[['Tweets']]

## Defining Labels for Annotation
candidate_labels = ["Positive", "Negative", "Neutral"]

## Model 2: RoBERTa (Pre- Trained zero- shot classification)
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")


## Function to annotate the statement

def annotate_statement_RoBERTa(statement):
    try:
        result = classifier(statement, candidate_labels=candidate_labels)

        ## Printing labels with scores
        # print(result)

        ## Label with highest Score
        annotation = result['labels'][0]
        return annotation
    except Exception as e:
        # print(e)
        return "Error"


data['Annotation'] = data['Tweets'].apply(annotate_statement_RoBERTa)

# print(data)
## Saving results to an excel file.
output_file = "annotated_statements_RoBERTa.xlsx"
data.to_excel(output_file, index=False)

