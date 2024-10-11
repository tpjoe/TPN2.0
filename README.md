# Information

## General folder structure (Samson, you can skip this)
- train.py: Trains a model in two phases - pretraining and finetuning.
- inference.py: Perform inference on csv input data and a single sample using a trained transformer model.
- model.py: Seq2SeqTransformer Network
- trainer.py: Helper functions for model training and inference
- dataset.py: MedicalDataset
- config.py: Configuration object containing model and operation settings
- utils.py: Helper functions for model training and inference
- parser.py: Parse command line arguments
- environment.py: Environment libraries to install.
- mock_data.py: Mock data for testing purposes.


## Usage
- To update the software database
```
python fetch.py --UID=N00001
```

- To preprocess the data, which includes handling nans, capping data range if exceed, encoding data, etc. Please have a look at config file for the input and output directory
```
python preprocess.py --todays_date=2024-08-21
python preprocess_TPN.py --todays_date=2024-08-21
```


- To make prediction
```
python get_recommendations.py --device=cuda:0 --rerank=False --n_clusters=5 --UID=N00001
```

This will return a dataframe of TPN composition from the top three choices. If rerank is True, you need to provide a dataframe of manual target (see /inputs/user_manual_target.csv). This code should work seamlessly on the preprocessed database.


- To get interpretation
```
python get_why.py --UID=N00001 --device=cuda:0
```

This will return a dataframe of the top feature importance in low, high, medium.

More details on the path default values are in the `config.py` file.


## Notes
- If any of the parsed arguments are `None`, it will falls back to the default values defined in `config.py`

## Database Structure and Flow
- Samson's Data Lake (DL) - Data Samson queried from Epic, these might have 2 labs updated at different timepoint of the same day
- DB - Our own database that aggregate data within the same day into the same row. This is an SQL database. Any missing values are retained here. The DB has 3 tables
    1. Person - This table contains static information of the patient, including UID, bw, gest_age, race_concept_id, and gender_concept_id
    2. Measurements - This table contains all other dynamic inputs from Samson's DL, such as lab values, TodaysWeight, day_since_birth, etc.
    3. TPNParams - This table contains all the TPN variables input by the physicians into the app.
- PDB - Processed database, all the same as DB structure, except the data in all 3 tables are processed and ready to the model.
- ADB - Adjustment database, only have 1 table and it collects all TPN target changes the user made. This is an input to rerank the choices.

![Flow Diagram](Flow.svg)

## To-do
- Improve speed of the model and SHAP calculations
- A function to allow flexible number of desired TPN2.0 formulas
- A function to rerank the choice given doctor's inputs
- Shrink and reorder redundant inputs
- Table for feature names to actual words


## Input Details
https://docs.google.com/spreadsheets/d/1rP8PXfDtwl67aSXsOiLZ5HUxOubqU9oEfHjvZdDh2sA/edit?gid=0#gid=0

| Parameter                    | Units            | 95% CI Range (NICU) |
|------------------------------|------------------|--------------|
| gest_age                      | Days             |  164-283     |
| bw                            | Kgs              |  0.5-4.0     |
| day_since_birth               | Days             |  0-157     |
| RxDay                         | Days             |  1-103     |
| TodaysWeight                  | Kgs              |  0.65-5.80     |
| TPNHours                      | Hours            |   24  |
| max_chole_TPNEHR              | -                |  1 if cholestasis     |
| Alb_lab_value                 | g/dL             |  1.7-3.7     |
| Ca_lab_value                  | mg/dL            |  7.4-11.0     |
| Cl_lab_value                  | mEq/L            |  92-113     |
| Glu_lab_value                 | mg/dL            |  61-182     |
| Na_lab_value                  | mEq/L            |  130-147     |
| BUN_lab_value                 | mg/dL            |  6-57     |
| Cr_lab_value                  | mg/dL            |  0.13-1.30     |
| Tri_lab_value                 | mg/dL            |  22-247     |
| ALKP_lab_value                | U/L              |  64-731     |
| CaI_lab_value                 | mg/dL            |  1.04-1.56     |
| CO2_lab_value                 | mEq/L            |  18-38     |
| PO4_lab_value                 | mg/dL            |  3.0-7.3     |
| K_lab_value                   | mEq/L            |  2.6-5.8     |
| Mg_lab_value                  | mg/dL            |  1.6-3.2     |
| AST_lab_value                 | U/L              |  16-258     |
| ALT_lab_value                 | U/L              |  6-224     |
| FluidDose                     | mL/kg/day        |  80-170     |
| VTBI                          | mL               |  55-626     |
| InfusionRate                  | mL/hr            |  2.29-26.38     |
| FatInfusionRate               | mL/hr            |  0.00-3.03     |
| ProtocolName_NEONATAL         | None             |  1 if NEONATAL     |
| ProtocolName_PEDIATRIC        | None             |  1 if PEDIATRIC     |
| LineID_1                      | None             |  1 if central     |
| LineID_2                      | None             |  1 if peripheral     |
| gender_concept_id_0           | None             |  1 if gender unknown     |
| gender_concept_id_8507        | None             |  1 if male     |
| gender_concept_id_8532        | None             |  1 if female     |
| race_concept_id_0             | None             |  1 if race unknown     |
| race_concept_id_8515          | None             |  1 if Asian     |
| race_concept_id_8516          | None             |  1 if Black     |
| race_concept_id_8527          | None             |  1 if White     |
| race_concept_id_8557          | None             |  1 if Native Hawaiian or Other Pacific Islander   |
| race_concept_id_8657          | None             |  1 if American Indian or Alaska Native     |
| FatProduct_SMOFlipid 20%      | None             |  1 if SMOFlipid 20%     |
| FatProduct_Intralipid 20%     | None             |  1 if Intralipid 20%     |
| FatProduct_Omegaven 10%       | None             |  1 if Omegaven 20%     |


