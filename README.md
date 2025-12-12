# Lab 02: HR Analytics: Job Change of Data Scientists

This project focuses on solving a binary classification problem in the field of **HR Analytics**. The core goal is to build a machine learning model that can predict the probability that a candidate will change jobs after completing a company training course. 

The unique feature and biggest challenge of the project is **"Pure NumPy Implementation"**. Instead of relying on high-level libraries such as Scikit-learn or Pandas, the entire process from data processing, matrix calculation to algorithm optimization is manually installed from scratch using the NumPy library.

---
## 📖 Table of Contents

- [Introduction](#-introduction)
- [Dataset](#-dataset)
- [Method](#-Method)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Challenges & Solutions](#-challenges_solutions)
- [Future Improvements](#-Future_improvements)
- [Credits](#-credits)
- [License](#-license)

---

## 🎯 Introduction
In the context of the volatile IT and Data Science human resources market, recruiting and retaining talent is a big challenge. A company specializing in training and recruiting Data Scientists is facing a problem: Many candidates, after completing the company's training course, decide to look for job opportunities elsewhere instead of staying to work.

- **Problem Formulation**: The problem is to build a machine learning model to predict the probability that a candidate will **change jobs** (Target = 1) or **no change jobs** (Target = 0) based on their demographic information and work history.

- **Motivation and Practical Applications**: 
    - **Cost Optimization**: Helps the recruiting department minimize costs and time spent on candidates who do not intend to stay long-term.

    - **Human resource risk management**: Allows the company to identify early groups of candidates with high risk of leaving so that appropriate retention policies can be developed.
    
    - **Improve training quality**: Analyze factors that influence students' decisions to improve the input selection process.

- **Goals**:
    - Build a complete Data Pipeline from **EDA** to **Preprocessing**, using the NumPy library.

    - Manually implement the **Machine Learning model** (Logistic Regression) algorithm and metrics, optimize with Vectorization to replace loops.

    - Achieve good prediction performance (stable F1-Score) on imbalanced datasets.
---

## 🗄️ Dataset

The dataset used is "HR Analytics: Job Change of Data Scientists", provided by Kaggle. Source: https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists
- **Size**: 19158 rows and 14 columns
- **Features**:
    - `enrollee_id` : Unique ID for candidate.
    - `city`: City code.
    - `city_development_index` : Developement index of the city (scaled).
    - `gender`: Gender of candidate
    - `relevent_experience`: Relevant experience of candidate
    - `enrolled_university`: Type of University course enrolled if any
    - `education_level`: Education level of candidate
    - `major_discipline` :Education major discipline of candidate
    - `experience`: Candidate total experience in years
    - `company_size`: No of employees in current employer's company
    - `company_type` : Type of current employer
    - `last_new_job`: Difference in years between previous job and current job
    - `training_hours`: training hours completed
    - `target`: 0 – Not looking for job change, 1 – Looking for a job change
- **Note**:
    - The dataset is imbalanced.
    - Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.
    - Missing imputation can be a part of your pipeline as well.

## 🧠 Method
The project follows standard data science processes, with an emphasis on implementing algorithms and processing techniques from scratch using the `NumPy` library, instead of relying on available functions from `Scikit-learn` or `Pandas`.

**Exploratory Data Analysis** (EDA)
- **Data Loading**:
- **Data Checking**:
    - Missing values
    - Validity check
    - Dupicates
- **Data Visualization**:
    - Univariate analysis
    - Multivariate analysis 

**Data Preprocessing**: 
- **Imputation**:
    - `Numerical` features: Fill in with the **Median** value
    - `Categorical`features: Fill in with the **Unknown** value
- **Encoding**: 
    - **Log Transformation** and **Standard Scaler** for `numerical` features
    - **Ordinary**, **One-hot** and **Frequency** for `categorical` features

**Model Building**:
The `Logistic Regression` model is built in Object Oriented Architecture (OOP) with core mathematical components:
- **Model Structure**:
    - **Activation Function**: Use the `Sigmoid` function to convert the linear output to probability ($0 \le P \le 1$).$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
    - **Weighted Loss Function** instead of using the standard Binary Cross-Entropy, I use a weighted version to penalize wrong predictions on the minority class (`Target=1`) more heavily.Formula:$$J(w) = - \frac{1}{m} \sum_{i=1}^{m} \alpha^{(i)} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

- **Optimizer**:
    - Use **Gradient Descent** to update weight $w$ and bias $b$
    - **Vectorization**: The entire prediction (Forward) and derivative (Backward) calculation process is done by matrix multiplication
    - **Forward Pass**: Calculate the prediction $\hat{y} = \sigma(X \cdot w + b)$
    - **Backward Pass** (Weighted Gradient): Calculate the derivative of the loss function taking into account the sample weights:$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\mathbf{\alpha} \odot (\hat{y} - y))$$
    - **Update Weights**: $$w := w - \eta \cdot \frac{\partial J}{\partial w}$$

- **Handling sample imbalance**:
    - Integrate the **Class Weight** (`'balanced'`) technique directly into the loss function to increase the penalty for mispredicting the minority class (`Target=1`) $$w_c = \frac{N_{samples}}{2 \times N_{class\_c}}$$

- **Review & Refine**:
    - **Metrics**: Manually implement `f1_score`, `precision`, `recall`, `confusion_matrix` functions using `NumPy`.
    - **Cross-Validation**: Manually build **K-Fold Cross-Validation** function to evaluate model stability.

---

## 💻  Tech Stack
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

- **Core Libraries:**
    - **[NumPy](https://numpy.org/):** Core Library. Used for all matrix calculations, data processing, and implementing machine learning algorithms from scratch.
    - **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/):** Used for data visualization (EDA) and model evaluation graphs.
---

## 🚀 Installation & Setup
To get the project running locally, follow these steps:

1.  **Clone the repository**
    ```sh
    git clone https://github.com/hnhan2005/Lab02_HR_Analysis.git
    cd .\Lab02_HR_Analysis\
    ```
2.  **Install the required dependencies**
    ```sh
    pip install -r requirements.txt
    ```
2.  **Open source code**
    ```sh
    code . 
    ```
---

## 🎮 How to Run

The project is designed as a **Sequential Pipeline**. The main processing source code is packaged in the `src/` folder, the Notebooks in the `notebooks/` folder will call the functions from `src` to execute.

Please run the Notebooks in the following **correct order** to ensure the correct Data Flow:

- **Step 1: Exploratory Data Analysis (EDA)** `notebooks/01_data_exploration.ipynb`

- **Step 2: Preprocecssing** `notebooks/02_preprocessing.ipynb`

- **Step 3: Modeling** `notebooks/03_modeling.ipynb`

## 📊 4. Results 

Details of the training and parameter tuning process can be found at: [📓 03_modeling.ipynb](notebooks/03_modeling.ipynb).

Below is a summary of the performance of the best **Logistic Regression** model on the test set.

### Model Performance
The model is optimized according to the **F1-Score** index to solve the problem of Imbalanced Data.

| Metric | Score | Ý nghĩa |
| :--- | :--- | :--- |
| **Accuracy** | **[0.7463]** | Overall accuracy. |
| **Precision** | **[0.4926]** | Of those predicted to leave, nearly [47]% actually leave.. |
| **Recall** | **[0.7663]** | The model found that nearly [75]% of people actually wanted to quit. |
| **F1-Score** | **[0.5997]** | The Balance Between Precision and Recall. |

> **Comment:** Although Accuracy is high, we focus on **Recall** to minimize missing out on departing talent.

### Confusion Matrix

![Confusion Matrix](/notebooks/confusion_matrix.png)

* **True Positives:** Successfully detected **[728]** employees intending to quit.
* **False Negatives:** No **[222]** employees detected.

### Top Feature Importance 
Based on the weights ($w$) of Logistic Regression, we find out the most influential factors:
![](/notebooks/features_importance.png)

### Business Insights 
* **High risk group:** Retention policy should focus on employees from low-growth cities and new graduates.
* **Training strategy:** Increasing Training Hours not only improves skills but is also an effective factor in retaining employees.

## 📂 Project Structure
```sh
LAB02_HR_ANALYSIS/
├── data/                                     
│   ├── raw/                                        # Raw data
│   │   ├── aug_train.csv                   
│   ├── processed/                                  # Processed data (after Preprocessing)
│   │   ├── X_train.npy  
│   │   ├── y_train.npy       
├── notebooks/                                      
│   ├── 01_data_exploration.ipynb                   # EDA Stage
│   ├── 02_preprocessing.ipynb                      # Preprocessing Stage
│   └── 03_modeling.ipynb                           # Modeling Stage
├── src/                                            # Processing source code for notebooks    
│   ├── __init__.py          
│   ├── data_processing.py                      
│   ├── models.py             
│   └── visualization.py      
├── README.md                 
└── requirements.txt          
```

## ⚠️ Challenges & Solutions
**Mixed data processing**:
- **Challenge**: `NumPy` is optimized for homogeneous arrays (`float, int`). HR Analytics data contains a mix of numbers and strings. Loading data using `np.genfromtxt` often results in formatting errors or data being converted to `NaN` if not configured properly.
- **Solution**:
    - Use Structured Arrays mode (`dtype=None`) when loading data to preserve the original data type.
    - Write separate processing functions: Separate processing flows for `numerical` and `categorical` variables.
    - Finally, cast the entire matrix to `float64` (astype(float)) only after coding is complete to ensure matrix calculations work.

**Computational Efficiency**:
- **Challenge**: Calculating distance or updating weights for a large number of data rows using a for loop (Python loop) will be extremely slow.
- **Solution**: Thoroughly apply **Vectorization** and **Broadcasting** thinking: 
    - Replace Python's `sum()` with `np.sum()`
    - Replace the loop that calculates `$z$` with the matrix multiplication `np.dot(X, w)`
    - Use `np.einsum` to calculate the sum of squared errors in T-test testing, helping to optimize memory and CPU speed.

**Numerical Stability**:
- **Challenge**: **Sigmoid** function contains the calculation $e^{-z}$. If $z$ is too large or too small (e.g. $z = -1000$), the computer will get an `OverflowError` or return `nan`.
- **Solution**: Use the **Clipping** technique: `z = np.clip(z, -250, 250)` before feeding it to the exponential function, ensuring that the value is always within the safe threshold without affecting the accuracy of the probability.

## ⭐ Future Improvements
The current project has achieved the basic goals of model building and code optimization. However, there is still a lot of room for improvement:

**Advanced Optimization**:
- Replace **Pure Gradient Descent** (Batch GD) with **Mini-batch Gradient Descent** or **Stochastic Gradient Descent** (SGD) to speed up convergence on large data.
- Implement advanced optimization algorithms like **Adam** or **RMSProp** in NumPy.

**Model Complexity**:
- Integrate **Regularization (L1 - Lasso / L2 - Ridge)** into the loss function to better control Overfitting when the number of features increases.
- Extend the model to **Softmax Regression** to support Multi-class Classification if the Target has more than 2 labels.

**Advanced Feature Enginering**:
- Experiment with **Polynomial Features** (create 2nd and 3rd order variables) to help linear models learn non-linear decision boundaries.
- Use **PCA (Principal Component Analysis)** self-coded in NumPy (using SVD) to reduce data dimensionality before training.

**Deployment**:
- Package the `LogisticRegressionNumpy` class into `pip` install Python library (Package).
- Build a simple API with Flask/FastAPI for real-time prediction.

## 👤 Credit
This project was created for the "Programming for Data Science" course lab.
- Author: Trầm Hữu Nhân
- ID: 23127442
- VNUHCM - University of Science