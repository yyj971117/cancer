# 乳腺癌分类项目

本项目旨在使用多个机器学习模型将乳腺癌肿瘤分类为恶性（M）或良性（B）。项目中使用的数据集为乳腺癌威斯康星（诊断）数据集。以下机器学习算法已实现并评估：逻辑回归、决策树、K-近邻（KNN）、随机森林和支持向量机（SVM）。

## 项目结构

- `data_loader.py`：包含加载和预处理数据的函数。
- `main.py`：包含运行项目所需的所有函数和类的主脚本。
- `models.py`：包含机器学习模型的实现。
- `requirements.txt`：所需的Python包。
- `utils.py`：工具函数。
- `visualization.py`：绘图和可视化函数。

## 安装与设置

1. 将存储库克隆到本地计算机：
    ```sh
    git clone https://github.com/yyj971117/cancer.git
    cd cancer
    ```

2. 确保已安装Python 3.7。

3. 使用pip安装所需的库：
    ```sh
    pip install -r requirements.txt
    ```

## 使用方法

1. 准备你的数据集。将数据集文件（`data.csv`）放在项目根目录下的`cancer`目录中。

2. 运行主脚本`main.py`：
    ```sh
    python main.py
    ```

运行脚本将执行以下步骤：

1. 加载并预处理数据。
2. 将数据分为训练集和测试集。
3. 训练和评估多个机器学习模型。
4. 保存结果，包括混淆矩阵、ROC曲线和预测CSV文件。

## 函数和类

### data_loader.py

- **load_and_preprocess_data(file_path)**：加载并预处理数据。

### utils.py

- **create_folder(folder_path)**：如果文件夹不存在则创建文件夹。
- **save_predictions_to_csv(y_true, y_pred, model_name, save_dir)**：将模型预测结果保存到CSV文件。
- **cross_validate_model(model, X, y, k=5)**：执行K折交叉验证。

### visualization.py

- **plot_confusion_matrix(cm, title, save_path)**：绘制混淆矩阵。
- **plot_roc_curve(fpr, tpr, roc_auc, name, save_path)**：绘制单个模型的ROC曲线。
- **plot_combined_roc_curve(roc_data, save_path)**：绘制多个模型的组合ROC曲线。
- **merge_images(image_paths, save_path, grid_size=(2, 3))**：将多张图片合并为一张图片。

### models.py

- **LogisticRegressionManual**：自定义实现的逻辑回归。
- **DecisionTreeManual**：自定义实现的决策树。
- **KNNManual**：自定义实现的K-近邻。
- **RandomForestManual**：自定义实现的随机森林。
- **SVMManual**：自定义实现的支持向量机。

## 结果展示

结果，包括混淆矩阵、ROC曲线和预测CSV文件，将保存到`cancer/tu`目录。以下是各模型的评估指标总结：

- **逻辑回归**：
  - 准确率：`log_reg_acc`
  - 精确率：`log_reg_prec`
  - 召回率：`log_reg_recall`
  - F1评分：`log_reg_f1`
  - AUC：`roc_auc_log_reg`
  - 交叉验证结果：`log_reg_cv_results`

- **决策树**：
  - 准确率：`decision_tree_acc`
  - 精确率：`decision_tree_prec`
  - 召回率：`decision_tree_recall`
  - F1评分：`decision_tree_f1`
  - AUC：`roc_auc_decision_tree`
  - 交叉验证结果：`decision_tree_cv_results`

- **K-近邻**：
  - 准确率：`knn_acc`
  - 精确率：`knn_prec`
  - 召回率：`knn_recall`
  - F1评分：`knn_f1`
  - AUC：`roc_auc_knn`
  - 交叉验证结果：`knn_cv_results`

- **随机森林**：
  - 准确率：`random_forest_acc`
  - 精确率：`random_forest_prec`
  - 召回率：`random_forest_recall`
  - F1评分：`random_forest_f1`
  - AUC：`roc_auc_random_forest`
  - 交叉验证结果：`random_forest_cv_results`

- **支持向量机**：
  - 准确率：`svm_acc`
  - 精确率：`svm_prec`
  - 召回率：`svm_recall`
  - F1评分：`svm_f1`
  - AUC：`roc_auc_svm`
  - 交叉验证结果：`svm_cv_results`

## 致谢

本项目使用的数据集为乳腺癌威斯康星（诊断）数据集。特别感谢UCI机器学习库提供了该数据集。

如有任何问题或疑问，请随时提交问题或联系项目维护者。

## 许可证

本项目采用MIT许可证。

---

感谢您使用乳腺癌分类项目！希望这能帮助您准确地分类乳腺癌。
