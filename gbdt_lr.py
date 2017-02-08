import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import copy
class FeatureTransformer:
    def __init__(self):
        self.lgbc = LGBMClassifier()
        self.lrc = LogisticRegression(penalty='l1')
        self.feature_importance = None

    def _update_feature_desciption(self, feature_description, feature, threshold, boundary):
        if feature in feature_description:
            feature_item = feature_description[feature]
            if boundary == 'left':
                feature_item[boundary] = max(feature_item[boundary], threshold) if boundary in feature_item else threshold
            elif boundary == 'right':
                feature_item[boundary] = min(feature_item[boundary], threshold) if boundary in feature_item else threshold
            elif boundary == 'is':
                feature_item[boundary] = threshold
            else:
                if boundary in feature_item:
                    feature_item[boundary].append(threshold)
                else:
                    feature_item[boundary] = [threshold]
        else:
            feature_item = {'name': self._feature_names[feature], boundary: threshold}
            if boundary == 'is not':
                feature_item[boundary] = [threshold]
            feature_description[feature] = feature_item


    def _resolve_tree(self, data, tree, index, feature):
        if 'leaf_index' in tree:
            self._leaves['indexes'].append(index.astype(int))
            self._leaves['features'].append(self._print_feature_description(feature))
        else:
            split_feature = tree['split_feature']
            decision_type = tree['decision_type']
            threshold = tree['threshold']
            left_child = tree['left_child']
            right_child = tree['right_child']

            left_feature = copy.deepcopy(feature)
            right_feature = feature

            if decision_type == 'no_greater':
                new_index = (data[:, split_feature] <= threshold).reshape((data.shape[0], 1))
                self._update_feature_desciption(left_feature, split_feature, threshold, 'right')
                self._update_feature_desciption(right_feature, split_feature, threshold, 'left')
            else:
                new_index = (data[:, split_feature] == threshold).reshape((data.shape[0], 1))
                self._update_feature_desciption(left_feature, split_feature, threshold, 'is')
                self._update_feature_desciption(right_feature, split_feature, threshold, 'is not')
            self._resolve_tree(data, left_child, np.logical_and(index, new_index), left_feature)
            self._resolve_tree(data, right_child, np.logical_and(index, np.logical_not(new_index)), right_feature)

    def _resolve_gbdt(self, gbdt, data):
        trees = gbdt['tree_info']
        self._leaves = {'indexes': [], 'features': []}
        self._feature_names = gbdt['feature_names']
        for tree in trees:
            self._resolve_tree(data, tree['tree_structure'], np.ones((data.shape[0], 1), dtype=bool), {})

        transformed_feature = np.concatenate(self._leaves['indexes'], axis=1)
        return transformed_feature, self._leaves['features']

    def _print_feature_description(self, feature_des):
        description = ''
        for _, feature_item in feature_des:
            feature_name = feature_item['name']
            if 'is' in feature_item:
                description += '[{} is {}]'.format(feature_name, feature_item['is'])
            elif 'is not' in feature_item:
                description += '[{} is not any of {}]'.format(feature_name, feature_item['is not'])
            else:
                left_bound = feature_item['left'] if 'left' in feature_item else '-inf'
                right_bound = feature_item['right'] if 'right' in feature_item else 'inf'
                description += '[{} < {} <= {}]'.format(left_bound, feature_name, right_bound)
        return description

    def fit(self, X_train, y_train, categorical_feature='auto'):
        self.lgbc.fit(X_train, y_train, categorical_feature=categorical_feature)
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
