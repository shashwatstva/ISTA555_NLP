Files Description:

1. initial_data.xlsx : original raw data of 150 records (statements).
2. human_annotated.xlsx: manually annotated data.
3. BART.py : Script implementing annotation using BART model. Takes initial_data as input files. Stores the results in "annotated_statements.xlsx".
4. RoBERTa.py: Script implementing annotation using RoBERTa model. Takes initial_data as input files. Stores the results in "annotated_statements_RoBERTa.xlsx".
5. BART_eval.py: Script calculating various evaluation matrices on the results of BART.py. Takes "human_annotated.xlsx" and "annotated_statements.xlsx" as input.
6. RoBERTA_eval.py: Script calculating various evaluation matrices on the results of BART.py. Takes "human_annotated.xlsx" and "annotated_statements_RoBERTa.xlsx" as input.
7. BART_Confusion.png and RoBERTa_Confusion.png: Confusion Matrix generated for BART and RoBERTa model respectively.


Run the files in the above order to avoid any errors.

Libraries used:
pandas
numpy
transformers
sklearn
seaborn
matplotlib
statsmodel
