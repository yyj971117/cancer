import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from data_loader import load_and_preprocess_data
from visualization import plot_confusion_matrix, plot_roc_curve, plot_combined_roc_curve, merge_images
from models import LogisticRegressionManual, DecisionTreeManual, KNNManual, RandomForestManual, SVMManual
from utils import create_folder, save_predictions_to_csv, cross_validate_model

def main():
    # 创建保存图片的文件夹
    create_folder(os.path.expanduser(r'/cancer\tu'))
    save_dir = os.path.expanduser(r'/cancer\tu')

    # 加载数据
    data = load_and_preprocess_data(os.path.expanduser(r'/cancer\data.csv'))

    # 分割特征
    features_mean = list(data.columns[1:11])
    features_se = list(data.columns[11:21])
    features_worst = list(data.columns[21:31])

    # 选择特征
    selected_features = features_mean[:5]  # 选择较少的特征，减少过拟合

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data[selected_features].to_numpy())
    y = data['diagnosis'].to_numpy()

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 存储各模型的ROC数据
    roc_data = []

    # 逻辑回归
    log_reg = LogisticRegressionManual(learning_rate=0.001, n_iters=2000, lambda_param=0.1)
    log_reg.fit(X_train, y_train)
    log_reg_cv_results = cross_validate_model(log_reg, X, y)
    y_pred_log_reg = log_reg.predict(X_test)
    log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
    log_reg_prec = precision_score(y_test, y_pred_log_reg, average='macro')
    log_reg_recall = recall_score(y_test, y_pred_log_reg, average='macro')
    log_reg_f1 = f1_score(y_test, y_pred_log_reg, average='macro')
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test))
    roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)
    roc_data.append((fpr_log_reg, tpr_log_reg, roc_auc_log_reg, 'Logistic Regression'))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_log_reg), 'Logistic Regression Confusion Matrix', os.path.join(save_dir, 'log_reg_cm.png'))
    plot_roc_curve(fpr_log_reg, tpr_log_reg, roc_auc_log_reg, 'Logistic Regression', os.path.join(save_dir, 'log_reg_roc.png'))
    save_predictions_to_csv(y_test, y_pred_log_reg, 'log_reg', save_dir)

    # 决策树
    decision_tree = DecisionTreeManual(max_depth=5)
    decision_tree.fit(X_train, y_train)
    decision_tree_cv_results = cross_validate_model(decision_tree, X, y)
    y_pred_decision_tree = decision_tree.predict(X_test)
    decision_tree_acc = accuracy_score(y_test, y_pred_decision_tree)
    decision_tree_prec = precision_score(y_test, y_pred_decision_tree, average='macro')
    decision_tree_recall = recall_score(y_test, y_pred_decision_tree, average='macro')
    decision_tree_f1 = f1_score(y_test, y_pred_decision_tree, average='macro')
    fpr_decision_tree, tpr_decision_tree, _ = roc_curve(y_test, decision_tree.predict_proba(X_test))
    roc_auc_decision_tree = auc(fpr_decision_tree, tpr_decision_tree)
    roc_data.append((fpr_decision_tree, tpr_decision_tree, roc_auc_decision_tree, 'Decision Tree'))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_decision_tree), 'Decision Tree Confusion Matrix', os.path.join(save_dir, 'decision_tree_cm.png'))
    plot_roc_curve(fpr_decision_tree, tpr_decision_tree, roc_auc_decision_tree, 'Decision Tree', os.path.join(save_dir, 'decision_tree_roc.png'))
    save_predictions_to_csv(y_test, y_pred_decision_tree, 'decision_tree', save_dir)

    # KNN
    knn = KNNManual(k=5)
    knn.fit(X_train, y_train)
    knn_cv_results = cross_validate_model(knn, X, y)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_prec = precision_score(y_test, y_pred_knn, average='macro')
    knn_recall = recall_score(y_test, y_pred_knn, average='macro')
    knn_f1 = f1_score(y_test, y_pred_knn, average='macro')
    fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test))
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    roc_data.append((fpr_knn, tpr_knn, roc_auc_knn, 'KNN'))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_knn), 'KNN Confusion Matrix', os.path.join(save_dir, 'knn_cm.png'))
    plot_roc_curve(fpr_knn, tpr_knn, roc_auc_knn, 'KNN', os.path.join(save_dir, 'knn_roc.png'))
    save_predictions_to_csv(y_test, y_pred_knn, 'knn', save_dir)

    # 随机森林
    random_forest = RandomForestManual(n_trees=10, max_depth=5)
    random_forest.fit(X_train, y_train)
    random_forest_cv_results = cross_validate_model(random_forest, X, y)
    y_pred_random_forest = random_forest.predict(X_test)
    random_forest_acc = accuracy_score(y_test, y_pred_random_forest)
    random_forest_prec = precision_score(y_test, y_pred_random_forest, average='macro')
    random_forest_recall = recall_score(y_test, y_pred_random_forest, average='macro')
    random_forest_f1 = f1_score(y_test, y_pred_random_forest, average='macro')
    fpr_random_forest, tpr_random_forest, _ = roc_curve(y_test, random_forest.predict_proba(X_test))
    roc_auc_random_forest = auc(fpr_random_forest, tpr_random_forest)
    roc_data.append((fpr_random_forest, tpr_random_forest, roc_auc_random_forest, 'Random Forest'))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_random_forest), 'Random Forest Confusion Matrix', os.path.join(save_dir, 'random_forest_cm.png'))
    plot_roc_curve(fpr_random_forest, tpr_random_forest, roc_auc_random_forest, 'Random Forest', os.path.join(save_dir, 'random_forest_roc.png'))
    save_predictions_to_csv(y_test, y_pred_random_forest, 'random_forest', save_dir)

    # SVM
    svm = SVMManual(learning_rate=0.0001, n_iters=2000, lambda_param=0.1)
    svm.fit(X_train, y_train)
    svm_cv_results = cross_validate_model(svm, X, y)
    y_pred_svm = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_prec = precision_score(y_test, y_pred_svm, average='macro')
    svm_recall = recall_score(y_test, y_pred_svm, average='macro')
    svm_f1 = f1_score(y_test, y_pred_svm, average='macro')
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.decision_function(X_test))
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    roc_data.append((fpr_svm, tpr_svm, roc_auc_svm, 'SVM'))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_svm), 'SVM Confusion Matrix', os.path.join(save_dir, 'svm_cm.png'))
    plot_roc_curve(fpr_svm, tpr_svm, roc_auc_svm, 'SVM', os.path.join(save_dir, 'svm_roc.png'))
    save_predictions_to_csv(y_test, y_pred_svm, 'svm', save_dir)

    # 合并 ROC 曲线
    plot_combined_roc_curve(roc_data, os.path.join(save_dir, 'combined_roc_curve.png'))

    # 输出各指标
    print(f'Logistic Regression Accuracy: {log_reg_acc:.2f}, Precision: {log_reg_prec:.2f}, Recall: {log_reg_recall:.2f}, F1 Score: {log_reg_f1:.2f}, AUC: {roc_auc_log_reg:.2f}')
    print(f'Logistic Regression Cross-Validation Results: {log_reg_cv_results}')

    print(f'Decision Tree Accuracy: {decision_tree_acc:.2f}, Precision: {decision_tree_prec:.2f}, Recall: {decision_tree_recall:.2f}, F1 Score: {decision_tree_f1:.2f}, AUC: {roc_auc_decision_tree:.2f}')
    print(f'Decision Tree Cross-Validation Results: {decision_tree_cv_results}')

    print(f'K-Nearest Neighbors Accuracy: {knn_acc:.2f}, Precision: {knn_prec:.2f}, Recall: {knn_recall:.2f}, F1 Score: {knn_f1:.2f}, AUC: {roc_auc_knn:.2f}')
    print(f'K-Nearest Neighbors Cross-Validation Results: {knn_cv_results}')

    print(f'Random Forest Accuracy: {random_forest_acc:.2f}, Precision: {random_forest_prec:.2f}, Recall: {random_forest_recall:.2f}, F1 Score: {random_forest_f1:.2f}, AUC: {roc_auc_random_forest:.2f}')
    print(f'Random Forest Cross-Validation Results: {random_forest_cv_results}')

    print(f'Support Vector Machine Accuracy: {svm_acc:.2f}, Precision: {svm_prec:.2f}, Recall: {svm_recall:.2f}, F1 Score: {svm_f1:.2f}, AUC: {roc_auc_svm:.2f}')
    print(f'Support Vector Machine Cross-Validation Results: {svm_cv_results}')

if __name__ == '__main__':
    main()
