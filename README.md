# Cirrhosis-Patient-Survival-Prediction
A machine learning project that classifies patients into **four cirrhosis stages (Stage 1–4)** using demographic, clinical, and laboratory features. The project focuses on data preprocessing, handling class imbalance, comparing multiple machine learning models, and improving model performance through hyperparameter and threshold tuning.

> **Note:** This project is developed for academic purposes and is not intended for medical diagnosis or clinical decision-making.

---

## Project Overview
Cirrhosis is a progressive liver disease that develops through several stages. This project aims to build a machine learning model that can classify the stage of cirrhosis based on patient clinical and laboratory data.

The dataset contains patient information such as:
* Demographic information
* Clinical signs including ascites, hepatomegaly, spider angiomas, and edema
* Laboratory measurements such as bilirubin, albumin, copper, alkaline phosphatase, SGOT, platelets, and prothrombin time

The project applies several classification models and evaluates their ability to distinguish between the four cirrhosis stages.

---

## Objectives
* Prepare and clean the cirrhosis dataset for machine learning.
* Handle missing values, inconsistent categorical values, and extreme values.
* Address class imbalance using **SMOTE**.
* Compare multiple classification algorithms.
* Perform hyperparameter tuning on the selected models.
* Apply threshold tuning to improve class-level performance, particularly recall.
* Analyze the performance of the resulting models across different cirrhosis stages.

---

## Dataset
The dataset initially contains **620 rows and 20 columns**.  

The target variable is: **`Stage`** — representing the cirrhosis stage from **1 to 4**.  

After removing records with missing target values, the dataset contains **612 observations**.

The initial class distribution is:
| Stage   | Number of Samples |
| ------- | ----------------: |
| Stage 1 |                21 |
| Stage 2 |                97 |
| Stage 3 |               311 |
| Stage 4 |               183 |

The dataset is highly imbalanced, with Stage 3 and Stage 4 containing substantially more observations than Stage 1 and Stage 2.

---

## Data Preprocessing
### 1. Removing Irrelevant Columns
The following columns were removed because they were not used as predictive features:
* `ID`
* `N_Days`
* `Status`
* `Drug`
* `Cholesterol`
* `Tryglicerides`

### 2. Removing Duplicate Records
Duplicate rows were identified and removed before further processing.

### 3. Handling the Target Variable
Rows with missing `Stage` values were removed, and the target was converted from floating-point values to integer labels.  The final target classes are: **1, 2, 3, and 4**

### 4. Train-Test Split
The dataset was divided into:
* **70% training data**
* **30% testing data**

Resulting in:
* Training set: **428 samples**
* Testing set: **184 samples**

### 5. Data Cleaning
Several inconsistencies were found in categorical variables. For example, the `Sex` column contained variations such as: `F`, `Female`, `M`, `Male`, and `Male `. These were standardized into `F` and `M`. Similarly, `Ascites` values such as `Yes`, `No`, `Y`, and `N` were standardized into `Y` and `N`

### 6. Age Transformation
The original `Age` feature was recorded in days. It was converted into years using: **Age in years = Age / 365**

### 7. Missing Value Handling
Missing values were handled separately for numerical and categorical variables.  
* **Numerical features:** Missing values were filled using the **median calculated from the training data**.  
* **Categorical features:** Missing values were filled using the **mode from the training data**.

### 8. Outlier Handling
Log transformation was applied to several numerical variables with extreme values:
* Bilirubin
* SGOT
* Platelets  
After transformation, scaling was applied to selected features.

### 9. Encoding
Categorical variables were converted into numerical representations.

Binary encoding was applied to:
* Sex
* Ascites
* Hepatomegaly
* Spiders

`Edema` was transformed using **Label Encoding**.

### 10. Class Imbalance
The training data showed a significant imbalance between cirrhosis stages.  

**SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

Before SMOTE:
| Stage | Samples |
| ----- | ------: |
| 1     |      17 |
| 2     |      66 |
| 3     |     215 |
| 4     |     130 |

After SMOTE:
| Stage | Samples |
| ----- | ------: |
| 1     |     200 |
| 2     |     220 |
| 3     |     215 |
| 4     |     130 |

This was used to provide the minority classes with more representation during model training.

---

## Machine Learning Models
Five classification algorithms were evaluated:
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **CatBoost**
5. **LightGBM**

The models were evaluated using:
* Accuracy
* Precision
* Recall
* F1-Score

Because the dataset contains imbalanced classes, **recall was given particular attention when analyzing model performance across the different cirrhosis stages**.

---

## Model Comparison
The initial model evaluation produced the following results:
| Model               |   Accuracy | Macro F1-Score |
| ------------------- | ---------: | -------------: |
| Logistic Regression |     62.50% |           0.40 |
| Random Forest       |     63.04% |           0.41 |
| XGBoost             |     64.13% |           0.47 |
| CatBoost            |     62.50% |           0.42 |
| LightGBM            | **65.22%** |       **0.49** |

Based on the initial evaluation, **LightGBM and XGBoost were selected for further optimization**.

---

## Hyperparameter Tuning
### LightGBM
GridSearchCV with **3-fold cross-validation** was used to search for better LightGBM parameters.  
The search included:
* `n_estimators`
* `learning_rate`
* `max_depth`
* `num_leaves`
* `min_child_samples`
* `subsample`
* `colsample_bytree`
* `reg_alpha`
* `reg_lambda`

The selected configuration achieved: **Accuracy: 63.04%**

### XGBoost
A similar GridSearchCV process was applied to XGBoost using:
* `n_estimators`
* `learning_rate`
* `max_depth`
* `subsample`
* `colsample_bytree`
* `gamma`
* `reg_alpha`
* `reg_lambda`

The tuned model achieved: **Accuracy: 61.96%**

---

## Threshold Tuning
After hyperparameter tuning, threshold tuning was explored for both **LightGBM and XGBoost**. The purpose was to investigate whether adjusting the probability threshold for individual classes could improve class-level performance, particularly **macro F1-score and recall**.

### LightGBM
The optimal thresholds identified for the four classes were:
| Stage   | Threshold |
| ------- | --------: |
| Stage 1 |      0.05 |
| Stage 2 |      0.25 |
| Stage 3 |      0.50 |
| Stage 4 |      0.15 |

The threshold-tuned LightGBM achieved:  
* **Accuracy: 65.76%**
* **Macro F1-Score: 0.49**  
Compared with the initial LightGBM model, threshold tuning improved the accuracy from **65.22% to 65.76%**.

### XGBoost
The threshold search produced:
| Stage   | Threshold |
| ------- | --------: |
| Stage 1 |      0.55 |
| Stage 2 |      0.20 |
| Stage 3 |      0.40 |
| Stage 4 |      0.45 |

The resulting threshold-tuned XGBoost achieved:  
* **Accuracy: 62.50%**
* **Macro F1-Score: 0.47**

---

## Results & Analysis
The project shows that model performance varies across the four cirrhosis stages, with the minority classes remaining more challenging to classify. LightGBM produced the highest initial accuracy and macro F1-score among the five evaluated models. Further threshold tuning slightly improved its overall accuracy and improved the detection of several classes.

The final LightGBM threshold-tuning results were:
| Metric         |     Result |
| -------------- | ---------: |
| Accuracy       | **65.76%** |
| Macro F1-Score |   **0.49** |

The class-level results were:
| Stage   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| Stage 1 |      0.25 |   0.25 |     0.25 |
| Stage 2 |      0.32 |   0.23 |     0.26 |
| Stage 3 |      0.80 |   0.73 |     0.77 |
| Stage 4 |      0.61 |   0.81 |     0.69 |

The results indicate that **Stage 3 and Stage 4 were classified more consistently than the minority stages**, while Stage 1 and Stage 2 remained more difficult to detect.
