import numpy as np
import pandas as pd
from datetime import datetime

#############
# Load data #
#############

train = pd.read_csv('train.csv')
X_submission = pd.read_csv('test.csv')

train.info()
train.describe()
train.head()

####################
# Train-Test split #
####################

y_label = 'pressure'
y = train[y_label]
train.drop(y_label, axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=81)

from sklearn.model_selection import train_test_split
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.2, random_state=81)


##########
# OPTUNA #
##########
import optuna
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error


def objective(trial):
    params = {
        'objective': 'l1',
        #'num_classes': 3,
        'metric': 'neg_mean_absolute_error',
        'verbosity': -1,
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'subsample_for_bin': 200000,
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1), # 0-0.2
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.4),
        'num_leaves': trial.suggest_int('num_leaves', 100, 3000),#100-2000
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.5),
    	'min_child_weight': trial.suggest_float('min_child_weight', 0.000, 0.100),
    	'min_child_samples': trial.suggest_int('min_child_samples', 100, 500),
    	'max_depth': trial.suggest_int('max_depth', 10, 30), #10-30
    	'max_bin': trial.suggest_int('max_bin', 100, 1000), #100-1000
    	'learning_rate': 0.1,
    	'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1), #0.1-1
    	#'num_threads': 16,
        'n_jobs': 1,
        'feature_pre_filter': False,
        'force_col_wise': True
    }
    
    model = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**params),
        transformer=StandardScaler()
    )
    model.fit(X_train, y_train)
    loss_func = mean_absolute_error(model.predict(X_test), y_test)

    return loss_func


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)
trial = study.best_trial
print("Best loss function value: {}".format(trial.value))
print(f"Best params: {trial.params}")


#############
# SCORE OUT #
#############

def score_out(X_pred, X_valid, y_valid, params):
    model = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**params, objective='mae'),
        transformer=StandardScaler()
    )
    print('Fitting the model...')
    model.fit(
        X_train, y_train.values.ravel(),
        eval_set = (X_valid, y_valid),
        early_stopping_rounds = 10, verbose=-1
    )
    preds = model.predict(X_pred)
    return preds


def create_output(df):
    df[['id', 'pressure']].to_csv(
        f'submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        index=False
    )
    print('done!')


# score out on the validation set
y_valid_pred, _, _ = score_out(X_valid, X_test, y_test, trial.params)
mean_absolute_error(y_valid_pred, y_valid)

# score out on the submission set
preds = score_out(X_submission, X_valid, y_valid, trial.params)
subm = pd.DataFrame(
    {
        'id': X_submission['id'].values,
        'pressure': preds
    }
)
create_output(subm)


###########
# Shapley #
###########

import shap
import matplotlib.pyplot as plt

model = lgb.LGBMRegressor(**trial.params, objective='mae')

model.fit(
    X_train, y_train.values.ravel(),
    eval_set = (X_valid, y_valid),
    early_stopping_rounds = 10, verbose=-1
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
shap.summary_plot(shap_values, X_valid, show=False)
plt.savefig('shap.png', dpi=150, bbox_inches='tight')

# raw feature importances
feature_importance_df = pd.DataFrame(
    {'var':model.feature_name_, 'importance':model.feature_importances_}
)