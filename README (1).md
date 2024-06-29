
# README

## 乳腺癌分类项目

本项目旨在使用多个机器学习模型将乳腺癌肿瘤分类为恶性（M）或良性（B）。项目中使用的数据集为乳腺癌威斯康星（诊断）数据集。以下机器学习算法已实现并评估：逻辑回归、决策树、K-近邻（KNN）、随机森林和支持向量机（SVM）。

### 项目结构

- `data.csv`：包含乳腺癌分类特征和标签的数据集。
- `main.py`：包含运行项目所需的所有函数和类的主脚本。
- `results/`：保存结果（包括混淆矩阵、ROC曲线和预测CSV文件）的目录。

### 安装与设置

1. 将存储库克隆到本地计算机。
2. 确保已安装Python 3.x。
3. 使用pip安装所需的库：
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn pillow
    ```

### 使用方法

运行`main.py`脚本将执行以下步骤：

1. 加载并预处理数据。
2. 将数据分为训练集和测试集。
3. 训练和评估多个机器学习模型。
4. 保存结果，包括混淆矩阵、ROC曲线和预测CSV文件。

使用以下命令运行脚本：

```bash
python main.py
```

### 函数和类

- **create_folder(folder_path)**：如果文件夹不存在则创建文件夹。
- **load_and_preprocess_data(file_path)**：加载并预处理数据。
- **plot_confusion_matrix(cm, title, save_path)**：绘制混淆矩阵。
- **plot_roc_curve(fpr, tpr, roc_auc, name, save_path)**：绘制单个模型的ROC曲线。
- **plot_combined_roc_curve(roc_data, save_path)**：绘制多个模型的组合ROC曲线。
- **save_predictions_to_csv(y_true, y_pred, model_name, save_dir)**：将模型预测结果保存到CSV文件。
- **LogisticRegressionManual**：自定义实现的逻辑回归。
- **DecisionTreeManual**：自定义实现的决策树。
- **KNNManual**：自定义实现的K-近邻。
- **RandomForestManual**：自定义实现的随机森林。
- **SVMManual**：自定义实现的支持向量机。
- **cross_validate_model(model, X, y, k=5)**：执行K折交叉验证。
- **merge_images(image_paths, save_path, grid_size=(2, 3))**：将多张图片合并为一张图片。

### 结果

结果，包括混淆矩阵、ROC曲线和预测CSV文件，将保存到`results/`目录。以下是各模型的评估指标总结：

- **逻辑回归**：
  - 准确率：`accuracy`
  - 精确率：`precision`
  - 召回率：`recall`
  - F1评分：`f1`
  - AUC：`roc_auc`
  - 交叉验证结果：`cv_results`

- **决策树**：
  - 准确率：`accuracy`
  - 精确率：`precision`
  - 召回率：`recall`
  - F1评分：`f1`
  - AUC：`roc_auc`
  - 交叉验证结果：`cv_results`

- **K-近邻**：
  - 准确率：`accuracy`
  - 精确率：`precision`
  - 召回率：`recall`
  - F1评分：`f1`
  - AUC：`roc_auc`
  - 交叉验证结果：`cv_results`

- **随机森林**：
  - 准确率：`accuracy`
  - 精确率：`precision`
  - 召回率：`recall`
  - F1评分：`f1`
  - AUC：`roc_auc`
  - 交叉验证结果：`cv_results`

- **支持向量机**：
  - 准确率：`accuracy`
  - 精确率：`precision`
  - 召回率：`recall`
  - F1评分：`f1`
  - AUC：`roc_auc`
  - 交叉验证结果：`cv_results`

### 致谢

本项目使用的数据集为乳腺癌威斯康星（诊断）数据集。特别感谢UCI机器学习库提供了该数据集。

如有任何问题或疑问，请随时提交问题或联系项目维护者。

### 许可证

本项目采用MIT许可证。

---

感谢您使用乳腺癌分类项目！希望这能帮助您准确地分类乳腺癌。
