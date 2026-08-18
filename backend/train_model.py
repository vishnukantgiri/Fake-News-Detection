import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================

fake = pd.read_csv("../dataset/Fake.csv")
true = pd.read_csv("../dataset/True.csv")

print("Fake articles:", len(fake))
print("Real articles:", len(true))


# ==========================================
# 2. ADD LABELS
# ==========================================

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. COMBINE DATA
# ==========================================

data = pd.concat(
    [fake, true],
    ignore_index=True
)


# ==========================================
# 4. CLEAN DATA
# ==========================================

data["title"] = data["title"].fillna("").astype(str)
data["text"] = data["text"].fillna("").astype(str)

# Remove extra spaces
data["title"] = data["title"].str.replace(r"\s+", " ", regex=True).str.strip()
data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()


# ==========================================
# 5. REMOVE EMPTY ARTICLES
# ==========================================

data = data[
    (data["title"].str.len() > 0) |
    (data["text"].str.len() > 20)
].copy()


# ==========================================
# 6. REMOVE DUPLICATES
# ==========================================

data["duplicate_check"] = (
    data["title"] + " " + data["text"]
).str.lower().str.strip()

before = len(data)

data = data.drop_duplicates(
    subset=["duplicate_check"]
).copy()

after = len(data)

print("\nDuplicates removed:", before - after)


# ==========================================
# 7. CREATE DIFFERENT TEXT VERSIONS
# ==========================================

# Full article
full_content = (
    data["title"] + " " + data["text"]
).str.strip()

# Text only
text_only = data["text"].str.strip()

# Title only
title_only = data["title"].str.strip()


# ==========================================
# 8. CREATE AUGMENTED DATASET
# ==========================================

full_df = pd.DataFrame({
    "content": full_content,
    "label": data["label"]
})

text_df = pd.DataFrame({
    "content": text_only,
    "label": data["label"]
})

title_df = pd.DataFrame({
    "content": title_only,
    "label": data["label"]
})


# Remove very short samples
text_df = text_df[
    text_df["content"].str.len() > 20
]

title_df = title_df[
    title_df["content"].str.len() > 10
]


# Combine all versions
augmented_data = pd.concat(
    [
        full_df,
        text_df,
        title_df
    ],
    ignore_index=True
)


# Remove empty values
augmented_data["content"] = (
    augmented_data["content"]
    .fillna("")
    .astype(str)
    .str.strip()
)

augmented_data = augmented_data[
    augmented_data["content"].str.len() > 10
]


print("\nOriginal articles:", len(data))
print("Training samples after augmentation:", len(augmented_data))


# ==========================================
# 9. INPUT / OUTPUT
# ==========================================

X = augmented_data["content"]
y = augmented_data["label"]


# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 11. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=150000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


print("\nCreating TF-IDF vectors...")

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print(
    "TF-IDF shape:",
    X_train_tfidf.shape
)


# ==========================================
# 12. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 13. TEST MODEL
# ==========================================

prediction = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    prediction
)


# ==========================================
# 14. MODEL RESULTS
# ==========================================

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction,
        target_names=[
            "FAKE",
            "REAL"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)


# ==========================================
# 15. SAVE MODEL
# ==========================================

print("\nSaving model...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# ==========================================
# 16. SUCCESS
# ==========================================

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")