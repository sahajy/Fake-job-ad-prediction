Fake Job Posting Prediction using Multilayer Perceptron (MLP)

Overview
This project aims to detect fraudulent job postings using a **Multilayer Perceptron (MLP)**, a type of artificial neural network. The model analyzes various features in job advertisements to classify them as either **real** or **fake**, helping job seekers avoid scams.

Key Features
MLP-based classification** for fake job detection.
- Text & metadata analysis** (e.g., job title, description, company profile).
- Data preprocessing** (handling missing values, text cleaning, feature engineering).
- Performance evaluation** using metrics like accuracy, precision, recall, and F1-score.
- Comparison with other ML models** (e.g., Logistic Regression, Random Forest).

 Dataset
- Source:** [Kaggle Fake Job Posting Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
- Features:  
  - Job title, description, requirements  
  - Company profile, location, industry  
  - Employment type, required experience, etc.  
- **Target Variable:** `fraudulent` (1 = Fake, 0 = Real)

 Tech Stack
- **Python** (Primary Language)  
- **Libraries:**  
  - `scikit-learn` (MLPClassifier, preprocessing, evaluation)  
  - `TensorFlow/Keras` (Alternative implementation)  
  - `pandas`, `numpy` (Data handling)  
  - `matplotlib`, `seaborn` (Visualization)  
  - `nltk`/`spaCy` (Text processing)  

 Implementation Steps
1. **Data Loading & Exploration**  
   - Check class imbalance, missing values, and key trends.  
2. **Preprocessing**  
   - Text cleaning (lowercasing, stopword removal, stemming).  
   - Feature extraction (TF-IDF, word embeddings).  
3. **Model Training**  
   - Train an MLP with hidden layers, dropout, and activation functions.  
   - Hyperparameter tuning (learning rate, epochs, batch size).  
4. **Evaluation**  
   - Confusion matrix, ROC-AUC, classification report.  
5. **Deployment** (Optional)  
   - Flask/Django API for real-time prediction.  
Results
- Achieved **XX% accuracy** on test data.  
- MLP outperformed [baseline model] due to its ability to capture non-linear patterns.  
