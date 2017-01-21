import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

class FeatureTransformer:
    def __init__(self):
        self.lgbc = LGBMClassifier()
        self.lrc = LogisticRegression(penalty='l1')
        self.feature_importance = None

    def _resolve_tree(self, data, tree, index, feature):
        if 'leaf_index' in tree:
            self._leaves['indexes'].append(index.astype(int))
            self._leaves['features'].append(feature)
        else:
            split_feature = tree['split_feature']
            decision_type = tree['decision_type']
            threshold = tree['threshold']
            left_child = tree['left_child']
            right_child = tree['right_child']
            if decision_type == 'no_greater':
                new_index = (data[:, split_feature] <= threshold).reshape((data.shape[0], 1))
                left_operator = '=<'
                right_operator = '>'
            else:
                new_index = (data[:, split_feature] == threshold).reshape((data.shape[0], 1))
                left_operator = '=='
                right_operator = '!='
            self._resolve_tree(data, left_child, np.logical_and(index, new_index), feature + '[{0} {1} {2}]'.format(self._feature_names[split_feature], left_operator, threshold))
            self._resolve_tree(data, right_child, np.logical_and(index, np.logical_not(new_index)), feature + '[{0} {1} {2}]'.format(self._feature_names[split_feature], right_operator, threshold))

    def _resolve_gbdt(self, gbdt, data):
        trees = gbdt['tree_info']
        self._leaves = {'indexes': [], 'features': []}
        self._feature_names = gbdt['feature_names']
        for tree in trees:
            self._resolve_tree(data, tree['tree_structure'], np.ones((data.shape[0], 1), dtype=bool), '')

        transformed_feature = np.concatenate(self._leaves['indexes'], axis=1)
        return transformed_feature, self._leaves['features']

    def fit(self, X_train, y_train):
        self.lgbc.fit(X_train, y_train)
        model = self.lgbc.booster_.dump_model()
        transformed_feature, feature_description = self._resolve_gbdt(model, X_train.values)
        self.lrc.fit(transformed_feature, y_train.values)
        transformed_feature_table = pd.DataFrame([self.lrc.coef_.flatten().tolist(), feature_description]).T
        transformed_feature_table.columns = ['weight', 'feature']
        transformed_feature_table['weight_abs'] = transformed_feature_table['weight'].abs()
        transformed_feature_table = transformed_feature_table.sort_values('weight_abs', ascending=False).drop(
            'weight_abs', axis=1)
        self.feature_importance = transformed_feature_table
        return self

    def predict(self, X_test):
        model = self.lgbc.booster_.dump_model()
        transformed_feature, feature_description = self._resolve_gbdt(model, X_test.values)
        return self.lrc.predict(transformed_feature)
