# 🫀 Heart Disease Prediction using PySpark (MLlib)

This project builds a binary classification model using **logistic regression** in PySpark to predict whether a patient is likely to have heart disease based on medical attributes.

## 📁 Dataset

- **Sources**: [Heart Disease Dataset]([(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset]) [Framingham Heart Study Dataset]([https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset])
- **Format**: CSV
- **Target Column**: `target` (1 = presence of heart disease, 0 = absence)

## 🚀 Technologies Used

- Apache Spark (PySpark)
- MLlib (Spark's machine learning library)
- Logistic Regression
- BinaryClassificationEvaluator

## 🛠️ Steps

1. **Initialize SparkSession**
2. **Load CSV data into a DataFrame**
3. **Preprocess**:
   - Handle nulls
   - Convert categorical columns with `StringIndexer`
   - Assemble features into a single vector with `VectorAssembler`
4. **Split** data into training and testing sets
5. **Train** a logistic regression model
6. **Evaluate**:
   - Show predictions
   - Calculate AUC using `BinaryClassificationEvaluator`

## 📊 Sample Output
| target | prediction | probability   |
| ------ | ---------- | ------------- |
| 1      | 1.0        | \[0.04, 0.96] |
| 0      | 0.0        | \[0.76, 0.24] |


## 📝 Usage

```bash
python3 heart-disease-prediction.py
Make sure Spark is configured and the dataset (heart.csv) is /user/root/
