import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import graphviz
from sklearn.preprocessing import LabelEncoder

class DecisionTree:
    def __init__(self, max_depth=None,min_samples_split=None):
        self.max_depth = max_depth
        self.tree = None
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # 如果满足停止生长的条件
        if (self.max_depth is not None and depth >= self.max_depth) or num_classes == 1 or \
            (self.min_samples_split is not None and num_samples < self.min_samples_split):
            return {'class': np.bincount(y).argmax(), 'num_samples': num_samples, 'num_classes': num_classes}

        # 选择最优切分
        best_split = self._find_best_split(X, y)

        if best_split is None:
            return {'class': np.bincount(y).argmax(), 'num_samples': num_samples, 'num_classes': num_classes}

        # 执行切分
        left_indices, right_indices, split_feature, split_value = best_split
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': split_feature, 'split_value': split_value,
                'left': left_subtree, 'right': right_subtree,
                'num_samples': num_samples, 'num_classes': num_classes}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float('inf')
        best_split = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            for value in feature_values:
                left_indices = np.where(X[:, feature_index] <= value)[0]
                right_indices = np.where(X[:, feature_index] > value)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (left_indices, right_indices, feature_index, value)

        return best_split

    def _gini_index(self, left_y, right_y):
        num_left = len(left_y)
        num_right = len(right_y)
        total = num_left + num_right
        p_left = num_left / total
        p_right = num_right / total
        gini_left = 1 - sum([(np.sum(left_y == c) / num_left) ** 2 for c in np.unique(left_y)])
        gini_right = 1 - sum([(np.sum(right_y == c) / num_right) ** 2 for c in np.unique(right_y)])
        gini_index = (p_left * gini_left) + (p_right * gini_right)
        return gini_index

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'class' in tree:
            return tree['class']
        if x[tree['feature_index']] <= tree['split_value']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])
        
    def post_prune(self, X_val, y_val):
        self.tree = self._post_prune_tree(X_val, y_val, self.tree)

    def _post_prune_tree(self, X_val, y_val, tree):
        if 'left' in tree and 'right' in tree:
            tree['left'] = self._post_prune_tree(X_val, y_val, tree['left'])
            tree['right'] = self._post_prune_tree(X_val, y_val, tree['right'])
            
            # 计算剪枝前的准确率
            y_pred = self.predict(X_val)
            accuracy_before_pruning = accuracy_score(y_val, y_pred)

            # 剪枝
            tree_copy = copy.deepcopy(tree)
            tree_copy.pop('left')
            tree_copy.pop('right')
            tree_copy.pop('feature_index')
            tree_copy['class']= np.bincount(y_val).argmax()
            #计算剪枝后的准确率
            y_pred_pruned = np.array([self._predict_tree(x, tree_copy) for x in X_val])
            accuracy_after_pruning = accuracy_score(y_val, y_pred_pruned)
            #若准确率不下降则剪枝
            if accuracy_after_pruning >= accuracy_before_pruning:
                return {'class': np.bincount(y_val).argmax(), 'num_samples': len(y_val), 'num_classes': len(np.unique(y_val))}
            else:
                return tree
        else:
            return tree


def visualize_tree(tree, feature_names=None, class_names=None):
    dot_data = tree_to_dot(tree, feature_names, class_names)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree", format="pdf", cleanup=True)
    return graph

def tree_to_dot(tree, feature_names=None, class_names=None):
    dot_data = 'digraph Tree {\nnode [shape=box] ;\n'
    dot_data += tree_to_dot_rec(tree, feature_names, class_names)
    dot_data += '}'
    return dot_data

def tree_to_dot_rec(tree, feature_names=None, class_names=None, node_id=0):
    if 'class' in tree:
        class_label = class_names[tree['class']] if class_names else str(tree['class'])
        return f'{node_id} [label="{class_label}", shape=ellipse];\n'
    else:
        feature_label = feature_names[tree['feature_index']] if feature_names else f'X{tree["feature_index"]}'
        if tree['split_value'] == 0:
            split_label = f'{feature_label}'
        else:
            split_label = f'{feature_label} <= {tree["split_value"]}'
        left_child_id = 2 * node_id + 1
        right_child_id = 2 * node_id + 2
        dot_data = f'{node_id} [label="{split_label}"] ;\n'
        dot_data += f'{node_id} -> {left_child_id} ;\n'
        dot_data += f'{node_id} -> {right_child_id} ;\n'
        dot_data += tree_to_dot_rec(tree['left'], feature_names, class_names, left_child_id)
        dot_data += tree_to_dot_rec(tree['right'], feature_names, class_names, right_child_id)
        return dot_data

# 加载数据集
train_url = "data/adult_train.txt"
test_url="data/adult_test.txt"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week','native-country','income']
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week','income']
df_train = pd.read_csv(train_url, names=columns, na_values='?', skipinitialspace=True)
df_test = pd.read_csv(test_url, names=columns, na_values='?', skipinitialspace=True)
# 删除不必要的属性
df_train.drop(columns=['native-country'], inplace=True)
df_test.drop(columns=['native-country'], inplace=True)
# 删除包含缺失值的行
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
# 将分类变量转换为数值变量
df_train['income'] = df_train['income'].map({'<=50K': 0, '>50K': 1})
df_test['income'] = df_test['income'].map({'<=50K.': 0, '>50K.': 1})

# 创建一个标签编码器
label_encoder = LabelEncoder()
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
# 对每个类别变量进行标签编码
combined_df = pd.concat([df_train, df_test], axis=0)
# 对每个类别变量进行标签编码
for col in categorical_cols:
    # 只对列中的值进行编码
    combined_df[col] = label_encoder.fit_transform(combined_df[col])
# 将编码后的数据拆分回训练集和测试集
df_train = combined_df[:len(df_train)]
df_test = combined_df[len(df_train):]
# 分离特征和标签
y_train = df_train['income']
X_train = df_train.drop('income', axis=1)
y_test=df_test['income']
X_test=df_test.drop('income',axis=1)

X_test_head=X_test.head()
y_test_head=y_test.head()

# 创建决策树模型
model = DecisionTree()
# 训练模型
model.fit(X_train.values, y_train.values)
graph = visualize_tree(model.tree,feature_names)
graph.view()
# 预测测试集
y_pred = model.predict(X_test.values)
y_pred_head=model.predict(X_test_head.values)
print(y_pred_head)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("cart决策树模型的准确率为: {:.2f}%".format(accuracy * 100))

# 计算精确度
precision = precision_score(y_test, y_pred)
print("精确度:", precision)

# 计算召回率
recall = recall_score(y_test, y_pred)
print("召回率:", recall)

# 计算F1分数
f1 = f1_score(y_test, y_pred)
print("F1分数:", f1)

# 计算AUC
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC:", roc_auc)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)

# 计算 Cohen's Kappa 系数
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa 系数:", kappa)

# 创建决策树模型
model = DecisionTree(max_depth=8,min_samples_split=10)
# 训练模型
model.fit(X_train.values, y_train.values)
model.post_prune(X_test.values,y_test.values)
y_pred_prune=model.predict(X_test.values)
graph = visualize_tree(model.tree,feature_names)
graph.view()
# 计算准确率
accuracy = accuracy_score(y_test, y_pred_prune)
print("cart决策树模型的准确率为: {:.2f}%".format(accuracy * 100))

# 计算精确度
precision = precision_score(y_test, y_pred_prune)
print("精确度:", precision)

# 计算召回率
recall = recall_score(y_test, y_pred_prune)
print("召回率:", recall)

# 计算F1分数
f1 = f1_score(y_test, y_pred_prune)
print("F1分数:", f1)

# 计算AUC
roc_auc = roc_auc_score(y_test, y_pred_prune)
print("AUC:", roc_auc)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_prune)
print("混淆矩阵:")
print(conf_matrix)

# 计算 Cohen's Kappa 系数
kappa = cohen_kappa_score(y_test, y_pred_prune)
print("Cohen's Kappa 系数:", kappa)