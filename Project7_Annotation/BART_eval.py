import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import seaborn as sns
import matplotlib.pyplot as plt

## Evaluating BART Model (model 1)
llm_data = pd.read_excel("annotated_statements.xlsx")
hdata = pd.read_excel("human_annotated.xlsx")
## Accuracy

llm_accuracy = accuracy_score(hdata['Annotation'], llm_data['Annotation'] )

print("BART Accuracy: ",llm_accuracy)

##Interannotator Agreement
## Cohen's Kappa

cohen_kappa = cohen_kappa_score(hdata['Annotation'], llm_data['Annotation'])
print("Cohen's Kappa for BART model: ", cohen_kappa)

##Precision, Recall, and F1-Score

print("BART Performance:")
report = classification_report(hdata['Annotation'], llm_data['Annotation'])
print(report)


## Confusion matrix for BART Model
cm_llm = confusion_matrix(hdata['Annotation'], llm_data['Annotation'])
sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(hdata['Annotation']), yticklabels=np.unique(hdata['Annotation']))
plt.title("Confusion Matrix for BART Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("done")