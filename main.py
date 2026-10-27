from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
corpus = {
    "Thor eating pizza,Loki is eating pizza,Ironman ate pizza already",
    "Apple is announcing new phone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating Biryani and you are eating grapes"
}
v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)
print(v.vocabulary_)
# Print idf of each word
all_feature_names = v.get_feature_names_out()
for word in all_feature_names:
    indx = v.vocabulary_.get(word)
    # Get score
    idf_score = v.idf_[indx]
    print(f"{word} : {idf_score}")
# Print transformed output from tf-idf
print(transform_output.toarray())
# Loading dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Tf-idf\Ecommerce_data.csv")
print(df.head())
print(df.label.value_counts())
print(df.shape)
df['label_num'] = df['label'].map({
    "Household" : 0,
    "Electronics" : 1,
    "Clothing & Accessories" : 2,
    "Books" : 3
})
print(df.head())
# Train test split
X_train,X_test,y_train,y_test  = train_test_split(df.Text,df.label_num,test_size=0.2,random_state=42)
# Tfidf vectorizer
tf = TfidfVectorizer()
X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)
# Classification model
clf = DecisionTreeClassifier()
clf.fit(X_train_tf,y_train)
y_pred = clf.predict(X_test_tf)
print(classification_report(y_test,y_pred))
# Testing on new data
msg = ["Contrast living Wooden Decorative box"]
msg_tf = tf.transform(msg)
print(clf.predict(msg_tf))