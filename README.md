# E-commerce Product Classification Using TF-IDF

## 📌 Project Overview

This project demonstrates how Natural Language Processing (NLP) and machine learning can automatically classify e-commerce products into predefined categories based on their text descriptions.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert textual data into numerical features and a **Decision Tree Classifier** to predict product categories.

The model is trained on an e-commerce dataset containing product descriptions and their corresponding labels.

## 🎯 Objectives

- Understand how TF-IDF converts text into numerical representations.
- Explore vocabulary generation and inverse document frequency (IDF) scores.
- Preprocess textual data for machine learning.
- Train a Decision Tree Classifier for text classification.
- Evaluate model performance using a classification report.
- Predict product categories from new product descriptions.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Dataset loading and manipulation
- **Scikit-learn** — Text vectorization, model training, data splitting, and evaluation
- **TF-IDF Vectorizer** — Text feature extraction
- **Decision Tree Classifier** — Supervised machine learning model

## 📂 Project Structure

```text
Tf-idf/
│
├── ecommerce_data.csv
├── main.py
└── README.md
```

*Note: Adjust the filenames to match your actual repository.*

## 📊 Dataset

The project uses `ecommerce_data.csv`, which contains text descriptions and product category labels.

The target categories are:

| Category | Encoded Label |
|---|---:|
| Household | 0 |
| Electronics | 1 |
| Clothing & Accessories | 2 |
| Books | 3 |

The `label` column contains the original category names, while `label_num` stores their numerical representations for model training.

## ⚙️ How It Works

### 1. TF-IDF Feature Extraction

TF-IDF converts text into numerical features that a machine learning model can process.

It combines two concepts:

- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Reduces the importance of terms that appear across many documents.

The project uses `TfidfVectorizer` to learn the vocabulary and calculate TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)
```

The vectorizer is fitted on the training data and then applied to the test data to avoid data leakage.

### 2. Data Preparation

The dataset is loaded using Pandas, and product category labels are mapped to numerical values.

```python
df['label_num'] = df['label'].map({
    "Household": 0,
    "Electronics": 1,
    "Clothing & Accessories": 2,
    "Books": 3
})
```

The dataset is divided into training and testing sets using an 80:20 split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['label_num'],
    test_size=0.2,
    random_state=42
)
```

### 3. Model Training

A Decision Tree Classifier learns patterns from the TF-IDF features and their corresponding product categories.

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train_tf, y_train)
```

### 4. Model Evaluation

The trained model predicts categories for the test dataset.

Its performance is evaluated using Scikit-learn's classification report.

```python
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test_tf)

print(classification_report(y_test, y_pred))
```

The report provides:

- **Precision:** The proportion of predicted instances of a category that are correct.
- **Recall:** The proportion of actual instances of a category that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Support:** The number of actual test samples belonging to each category.

### 5. Predicting New Product Categories

The trained model can classify previously unseen product descriptions.

```python
msg = ["Contrast living Wooden Decorative box"]

msg_tf = tf.transform(msg)

prediction = clf.predict(msg_tf)

print(prediction)
```

The predicted numerical label corresponds to one of the four product categories.

To display the category name instead, use:

```python
label_mapping = {
    0: "Household",
    1: "Electronics",
    2: "Clothing & Accessories",
    3: "Books"
}

predicted_category = label_mapping[prediction[0]]

print("Predicted category:", predicted_category)
```

## 📈 Results

The model's performance is assessed using a classification report generated from the test dataset.

The report measures precision, recall, and F1-score for each product category.

*Insert your actual classification report here after running the model.*

## 🧠 Key Learning Outcomes

Through this project, I explored:

- Text preprocessing and feature extraction concepts.
- Vocabulary generation using `TfidfVectorizer`.
- The relationship between term frequency and inverse document frequency.
- Converting unstructured text into numerical machine learning features.
- Training a supervised classification model.
- Evaluating multiclass classification performance.
- Applying a trained model to new text inputs.
- Preventing data leakage by fitting the vectorizer only on training data.

## 🚀 Potential Improvements

Future improvements could include:

- Comparing Decision Trees with Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
- Evaluating models using cross-validation.
- Tuning hyperparameters to improve generalization.
- Investigating class imbalance in the dataset.
- Adding text preprocessing and feature engineering where appropriate.
- Building a Streamlit or Flask application for interactive product classification.
- Tracking experiments and comparing model performance.

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

Interested in Machine Learning, Artificial Intelligence, Data Science, and practical AI applications.

- GitHub: [Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

## 📜 License

This project is intended for educational and learning purposes.
