# Information

## General info
This repository hosts the codes for the TPN2.0 models, including the VAE-based and transformer-based ones. The code can be run on a shortlisted mock data provided in the mock_data folder. It will return the model performance based on the mock data, which should be close to 0 as the mock data were based on random distributions.

## Usage
- To run the VAE-based model
```
python main_VAE.py
```

- To run the transformer-based model
```
python transformer.py
```


## Input Details

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
| ProtocolName_NEONATAL         | None             |  1 if NEONATAL; 0 means PEDIATRIC     |
| LineID_2                      | None             |  1 if peripheral; 0 means central     |
| gender_concept_id_8507        | None             |  1 if male     |
| gender_concept_id_8532        | None             |  1 if female     |
| race_concept_id_8515          | None             |  1 if Asian     |
| race_concept_id_8516          | None             |  1 if Black     |
| race_concept_id_8527          | None             |  1 if White     |
| race_concept_id_8557          | None             |  1 if Native Hawaiian or Other Pacific Islander   |
| race_concept_id_8657          | None             |  1 if American Indian or Alaska Native     |
| FatProduct_SMOFlipid 20%      | None             |  1 if SMOFlipid 20%     |
| FatProduct_Intralipid 20%     | None             |  1 if Intralipid 20%     |
| FatProduct_Omegaven 10%       | None             |  1 if Omegaven 20%     |
| Encoded_0 (0-32)       | None             |  Encoded EHR representation from AE (32 dimensions)     |


