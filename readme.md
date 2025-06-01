# RedditEngage (CSP - Comment Score Predictor)

A distributed machine learning pipeline to predict Reddit comment popularity based on metadata and content, built using Apache Spark, Google Cloud, and Scikit-learn.

## 📦 Dataset

We use a curated 23 GB subset of the "Reddit Comments/Submissions (2005–2024)" dataset (3.12TB), covering Jan 2009 to Dec 2011 (~37.6M comments). The data is stored in JSON format and hosted on Google Cloud Storage.

## ⚙️ Environment

- Apache Spark (Google Dataproc)
- Google Cloud CLI
- Python 3.x
- Jupyter Notebook
- Scikit-learn

## 📊 Methodology

### 1. Data Ingestion & Preprocessing
- Load JSON from GCS into Spark
- Remove `[deleted]`, `[removed]` comments & those with missing `ups`
- Feature engineering: subreddit, time of post, comment length, edit delay
- Binary classification label: **Popular = ≥5 upvotes**

### 2. Feature Extraction
- `RegexTokenizer` → `HashingTF` → `IDF` for text
- OneHot encode subreddit
- VectorAssembler for combining all features

### 3. Models
- **Spark MLlib**: Logistic Regression & Naive Bayes
- **From Scratch**: SGD-based Logistic Regression using RDDs
- **Scikit-learn (Driver Node)**: Logistic Regression + Naive Bayes with SHAP analysis and coefficient plots

### 4. Evaluation
- Metrics: Accuracy, Precision, Recall, F1, AUC
- Train-Test Split: 85/15

## 📈 Results

| Model                   | Accuracy | F1 Score | AUC   |
|------------------------|----------|----------|-------|
| Spark Logistic Reg     | 0.625    | 0.448    | 0.689 |
| Spark Naive Bayes      | 0.601    | 0.410    | 0.654 |
| Custom SGD LogisticReg | 0.613    | 0.433    | 0.674 |
| Sklearn Logistic Reg   | 0.640    | 0.450    | 0.706 |
| Sklearn Naive Bayes    | 0.618    | 0.417    | 0.669 |

## 🔍 Insights

- Polite and positive language boosts upvotes.
- Toxic, dismissive tone leads to downvotes.
- r/science and r/AskReddit are more predictable than r/funny.
- SHAP & coefficient analysis helped validate key features.

## 🧪 How to Run

1. Clone this repo
2. Set up a Google Cloud project and Dataproc cluster
3. Upload dataset to GCS
4. Run Spark jobs via `spark-submit` or notebooks
5. For Sklearn, sample data using `.sample(fraction=0.1)` from Spark to local node

## 📂 Project Structure

📁 redditengage/
├── 📄 spark_pipeline.py # MLlib training pipeline
├── 📄 sgd_from_scratch.py # Custom logistic regression
├── 📄 sklearn_diagnostics.ipynb # SHAP and coefficient analysis
├── 📄 data_preprocessing.py # Cleaning & feature engineering
├── 📁 data/ # JSON dataset (stored in GCS)
└── 📁 plots/ # Coefficient barplots, ROC curves

## 🤝 Contributors

- Athul Thulasidasan
- Benyamin Tafreshian

## 📜 License

MIT License – see `LICENSE` file for details.
