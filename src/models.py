import numpy as np

# ==========================================
# MATH & UTILS
# ==========================================

def sigmoid(z):
    z = np.clip(z, -250, 250)

    return 1 / (1 + np.exp(-z))

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ==========================================
# (METRICS - NUMPY ONLY)
# ==========================================

def get_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    TP, _, FP, _ = get_confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall_score(y_true, y_pred):
    TP, _, _, FN = get_confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

# ==========================================
# CLASS LOGISTIC REGRESSION 
# ==========================================

class LogisticRegressionNumpy:
    def __init__(self, learning_rate=0.01, n_iterations=1000, class_weight=None):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = [] 

        self.class_weight = class_weight

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        sample_weights = np.ones(n_samples)

        if self.class_weight == 'balanced':
            n_0 = np.sum(y == 0)
            n_1 = np.sum(y == 1)
            
            w_0 = n_samples / (2 * n_0)
            w_1 = n_samples / (2 * n_1)
            
            sample_weights = np.where(y == 1, w_1, w_0)

        for i in range(self.n_iters):
            # Forward Pass: Calculate z = wX + b and y_hat = sigmoid(z)
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            # Backward Pass: Calculate Gradient (Derivative)
            error = y_pred - y
            weighted_error = error * sample_weights 
            dw = (1 / n_samples) * np.dot(X.T, weighted_error)
            db = (1 / n_samples) * np.sum(weighted_error)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Logging
            if i % 100 == 0:
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                # cost = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
                loss_per_sample = sample_weights * (y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
                cost = -np.mean(loss_per_sample)
                self.cost_history.append(cost)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.3):
        y_proba = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_proba])

# ==========================================
# CROSS VALIDATION 
# ==========================================

def k_fold_cross_validation(X, y, k=5, lr=0.1, epochs=2000):
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    scores = []

    for i in range(k):
        
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]

        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
      
        model = LogisticRegressionNumpy(learning_rate=lr, n_iterations=epochs)
        model.fit(X_train_fold, y_train_fold) 

        y_pred = model.predict(X_val_fold)
        score = f1_score(y_val_fold, y_pred)
        scores.append(score)
        
        print(f"Fold {i+1}/{k}: F1-Score = {score:.4f}")
        
    avg_score = np.mean(scores)
    print(f"=> Average F1-Score: {avg_score:.4f}")
    
    return scores