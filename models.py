import numpy as np
from collections import Counter

class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01, random_seed=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        np.random.seed(random_seed)
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.lambda_param * self.w
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_model)

class DecisionTreeManual:
    def __init__(self, max_depth=10, random_seed=42):
        self.max_depth = max_depth
        self.tree = None
        np.random.seed(random_seed)

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_proba(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth < self.max_depth and n_samples >= 2:
            best_idx, best_thr = self._best_split(X, y, n_features)
            if best_idx is not None:
                left_idxs, right_idxs = self._split(X[:, best_idx], best_thr)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    return Leaf(np.bincount(y).argmax())
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_idx, best_thr, left, right)
        return Leaf(np.bincount(y).argmax())

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thr = None, None
        for idx in range(n_features):
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thr = thr
        return split_idx, split_thr

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if isinstance(node, Leaf):
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _traverse_proba(self, x, node):
        if isinstance(node, Leaf):
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)

class KNNManual:
    def __init__(self, k=5, random_seed=42):
        self.k = k
        np.random.seed(random_seed)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict_proba(self, X):
        y_prob = []
        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            count = Counter(k_nearest_labels)
            y_prob.append(count[1] / self.k)
        return np.array(y_prob)

class RandomForestManual:
    def __init__(self, n_trees=10, max_depth=10, random_seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        np.random.seed(random_seed)

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(y), len(y), replace=True)
            tree = DecisionTreeManual(max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        tree_probs = np.swapaxes(tree_probs, 0, 1)
        y_prob = [np.mean(tree_prob) for tree_prob in tree_probs]
        return np.array(y_prob)

class SVMManual:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, random_seed=42):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        np.random.seed(random_seed)
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * (y_[idx] if not condition else 0)

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

class Node:
    def __init__(self, feature_idx, threshold, left, right):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, value):
        self.value = value

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
