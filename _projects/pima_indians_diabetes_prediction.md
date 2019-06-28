---
title: "Pima Indians Diabetes Prediction"
excerpt: "test"
---

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

### Read Data


```python
diabetes_data = pd.read_csv("pima-indians-diabetes-database/diabetes.csv")
diabetes_data = diabetes_data.astype(float)
```

### Exploratory Data Analysis


```python
# View of Data

diabetes_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>148.0</td>
      <td>72.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>85.0</td>
      <td>66.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>137.0</td>
      <td>40.0</td>
      <td>35.0</td>
      <td>168.0</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Column Stats

diabetes_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Look for Null Values

diabetes_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null float64
    Glucose                     768 non-null float64
    BloodPressure               768 non-null float64
    SkinThickness               768 non-null float64
    Insulin                     768 non-null float64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null float64
    Outcome                     768 non-null float64
    dtypes: float64(9)
    memory usage: 54.1 KB
    


```python
# Histograms

diabetes_data.hist(figsize=(10, 15), edgecolor = 'black', grid = False, bins = 20)
plt.show()
```


#![](output_7_0.png)



```python
# Pair Plots 

diabetes_data_noNA = diabetes_data.dropna()

sns.pairplot(diabetes_data_noNA, hue = 'Outcome', vars = ['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', \
                                                    'Glucose', 'Insulin', 'Pregnancies'], dropna = True)
plt.show()
```


#![png](output_8_0.png)



```python
# Correlation Matrix

plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_data[diabetes_data.columns[:8]].corr(), annot=True)
plt.show()
```


#![png](output_9_0.png)


### Basic Model Implementation


```python
# Create training and test data

from sklearn.model_selection import train_test_split

X_train_init, X_test_init = train_test_split(diabetes_data, test_size = 0.2, random_state = 1)

y_train = X_train_init.Outcome
X_train = X_train_init.drop("Outcome", axis = 1)

y_test = X_test_init.Outcome
X_test = X_test_init.drop("Outcome", axis = 1)
```


```python
# Try Different Models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


models = [LogisticRegression(solver = 'liblinear'), GaussianNB(), \
          RandomForestClassifier(n_estimators = 100, max_depth = 5), \
          ExtraTreesClassifier(n_estimators = 100, max_depth = 5), XGBClassifier()]

model_names = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Extra Trees', 'XGBoost']

k_folds = 5
```


```python
print("Cross validation:")
for i, model in enumerate(models):
    scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='accuracy')
    print(model_names[i], ": ", scores.mean())
```

    Cross validation:
    Logistic Regression :  0.765487568413
    Naive Bayes :  0.749253201945
    Random Forest :  0.757357057177
    Extra Trees :  0.742669168892
    XGBoost :  0.737778222045
    


```python
from sklearn import metrics

print("Test Set:")
for i, model in enumerate(models):
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print(model_names[i], ": ", accuracy)
```

    Test Set:
    Logistic Regression :  0.779220779221
    Naive Bayes :  0.772727272727
    Random Forest :  0.792207792208
    Extra Trees :  0.746753246753
    XGBoost :  0.798701298701
    

### Impute Missing Values


```python
X_train_init_null = X_train_init.copy()
X_test_init_null = X_test_init.copy()

X_train_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    X_train_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    
X_test_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    X_test_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    

#X_train_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = \
#X_train_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].apply(np.log)

#X_test_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = \
#X_test_init_null[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].apply(np.log)
```


```python
# Impute missing values

diabetes_median = X_train_init_null[X_train_init_null.Outcome == 1]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].median()
    
diabetes_mean = X_train_init_null[X_train_init_null.Outcome == 1]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].mean()
    
diabetes_std = X_train_init_null[X_train_init_null.Outcome == 1]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].std()
    

no_diabetes_median = X_train_init_null[X_train_init_null.Outcome == 0]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].median()
    
no_diabetes_mean = X_train_init_null[X_train_init_null.Outcome == 0]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].mean()
    
no_diabetes_std = X_train_init_null[X_train_init_null.Outcome == 0]\
    [['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].std()
    
    
print("Median of People with Diabetes:")
print(diabetes_median, '\n')

print("Mean of People with Diabetes:")
print(diabetes_mean, '\n')

print("Standard Deviation of People with Diabetes:")
print(diabetes_std, '\n')


print("Median of People without Diabetes:")
print(no_diabetes_median, '\n')

print("Mean of People without Diabetes:")
print(no_diabetes_mean, '\n')

print("Standard Deviation of People without Diabetes:")
print(no_diabetes_std, '\n')

    
print("Difference Between Medians of People with and without Diabetes:")
print(diabetes_median - no_diabetes_median, '\n')

print("Difference Between Mean of People with and without Diabetes:")
print(diabetes_mean - no_diabetes_mean)

diabetes_median_array = np.array(diabetes_median)
no_diabetes_median_array = np.array(no_diabetes_median)

diabetes_mean_array = np.array(diabetes_mean)
no_diabetes_mean_array = np.array(no_diabetes_mean)

diabetes_std_array = np.array(diabetes_std)
no_diabetes_std_array = np.array(no_diabetes_std)
```

    Median of People with Diabetes:
    Glucose          140.0
    BloodPressure     74.0
    SkinThickness     32.0
    Insulin          175.0
    BMI               34.3
    dtype: float64 
    
    Mean of People with Diabetes:
    Glucose          142.032864
    BloodPressure     74.828283
    SkinThickness     32.338129
    Insulin          212.466019
    BMI               35.074408
    dtype: float64 
    
    Standard Deviation of People with Diabetes:
    Glucose           29.710773
    BloodPressure     12.861654
    SkinThickness     10.307978
    Insulin          132.670225
    BMI                6.465322
    dtype: float64 
    
    Median of People without Diabetes:
    Glucose          107.0
    BloodPressure     70.0
    SkinThickness     27.0
    Insulin           95.0
    BMI               30.1
    dtype: float64 
    
    Mean of People without Diabetes:
    Glucose          111.052764
    BloodPressure     70.968992
    SkinThickness     27.024648
    Insulin          129.970732
    BMI               30.667677
    dtype: float64 
    
    Standard Deviation of People without Diabetes:
    Glucose           25.086143
    BloodPressure     12.463587
    SkinThickness      9.941672
    Insulin          105.244250
    BMI                6.572938
    dtype: float64 
    
    Difference Between Medians of People with and without Diabetes:
    Glucose          33.0
    BloodPressure     4.0
    SkinThickness     5.0
    Insulin          80.0
    BMI               4.2
    dtype: float64 
    
    Difference Between Mean of People with and without Diabetes:
    Glucose          30.980100
    BloodPressure     3.859291
    SkinThickness     5.313482
    Insulin          82.495288
    BMI               4.406731
    dtype: float64
    


```python
X_train_null = X_train_init_null.drop("Outcome", axis = 1)
X_test_null = X_test_init_null.drop("Outcome", axis = 1)
```


```python
def impute_nan(input_row):
    
    input_row_short = input_row[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    missing_indexes = np.argwhere(np.isnan(input_row_short))

    row_no_nan = np.array([i for j, i in enumerate(input_row_short) if j not in missing_indexes])
    diabetes_median_no_nan = np.array([i for j, i in enumerate(diabetes_median_array) if j not in missing_indexes])
    no_diabetes_median_no_nan = np.array([i for j, i in enumerate(no_diabetes_median_array) if j not in missing_indexes])
    
    euclidean_diabetes = np.linalg.norm(row_no_nan - diabetes_median_no_nan)
    euclidean_no_diabetes = np.linalg.norm(row_no_nan - no_diabetes_median_no_nan)
    
    min_dist = min(euclidean_diabetes, euclidean_no_diabetes)
    
    if min_dist == euclidean_diabetes:
        for val in missing_indexes:
            input_row_short[val] = diabetes_median_array[val[0]]
            #input_row_short[val] = np.random.normal(diabetes_mean_array[val[0]], diabetes_std_array[val[0]], 1)
    else:
        for val in missing_indexes:
            input_row_short[val] = no_diabetes_median_array[val[0]]
            #input_row_short[val] = np.random.normal(no_diabetes_mean_array[val[0]], no_diabetes_std_array[val[0]], 1)
            
    input_row[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = input_row_short
            
    return input_row
```


```python
X_train_imputed = X_train_null.copy()
X_test_imputed = X_test_null.copy()

X_train_imputed = X_train_imputed.apply(impute_nan, axis = 1)
X_test_imputed = X_test_imputed.apply(impute_nan, axis = 1)
```


```python
print("Cross validation:")
for i, model in enumerate(models):
    scores = cross_val_score(model, X_train_imputed, y_train, cv=k_folds, scoring='accuracy')
    print(model_names[i], ": ", scores.mean())
```

    Cross validation:
    Logistic Regression :  0.752491906463
    Naive Bayes :  0.74273602387
    Random Forest :  0.758877094324
    Extra Trees :  0.745921846315
    XGBoost :  0.750773454059
    

### New Features


```python
X_train_addedColumns = X_train_imputed.copy()
X_test_addedColumns = X_test_imputed.copy()
```


```python
### Add new features

def add_features(initial_df):  
    # Obesity
    initial_df.loc[:,'Obesity'] = 'None'
    initial_df.loc[(initial_df['BMI'] <= 18.5), 'Obesity'] = 'Underweight'
    initial_df.loc[(initial_df['BMI'] >= 18.5) & \
                             (initial_df['BMI'] < 25), 'Obesity'] = 'Normal'
    initial_df.loc[(initial_df['BMI'] >= 25) & \
                             (initial_df['BMI'] < 30), 'Obesity'] = 'Overweight'
    initial_df.loc[(initial_df['BMI'] >= 30), 'Obesity'] = 'Obese'

    # Blood Pressure
    initial_df.loc[:, 'Hypertension'] = 'None'
    initial_df.loc[(initial_df['BloodPressure'] < 80), 'Hypertension'] = 'Normal or Elevated'
    initial_df.loc[(initial_df['BloodPressure'] >= 80) & \
                             (initial_df['BloodPressure'] < 90), 'Hypertension'] = 'Hypertension Stage 1'
    initial_df.loc[(initial_df['BloodPressure'] >= 90), 'Hypertension'] = 'Hypertension Stage 2'

    # OGTT test
    initial_df.loc[:, 'OGTT'] = 'None'
    initial_df.loc[(initial_df['Glucose'] < 140), 'OGTT'] = 'Normal'
    initial_df.loc[(initial_df['Glucose'] >= 140) & \
                             (initial_df['Glucose'] < 200), 'OGTT'] = 'Pre-Diabetes'
    initial_df.loc[(initial_df['Glucose'] >= 200), 'OGTT'] = 'Diabetes'

    # Polynomials
    initial_df['Glucose_squared'] = initial_df['Glucose'] ** 2
    initial_df['Glucose_cubed'] = initial_df['Glucose'] ** 3

    initial_df['BMI_squared'] = initial_df['BMI'] ** 2
    initial_df['BMI_cubed'] = initial_df['BMI'] ** 3

    initial_df['DiabetesPedigreeFunction_squared'] = initial_df['DiabetesPedigreeFunction'] ** 2
    initial_df['DiabetesPedigreeFunction_cubed'] = initial_df['DiabetesPedigreeFunction'] ** 3

    # Interactions
    initial_df['Pregnancies_x_Age'] = initial_df['Pregnancies'] * initial_df['Age']
    initial_df['BP_x_Glucose'] = initial_df['BloodPressure'] * initial_df['Glucose']
    initial_df['DPF_x_Glucose'] = initial_df['DiabetesPedigreeFunction'] * initial_df['Glucose']
    initial_df['BMI_x_Age'] = initial_df['BMI'] * initial_df['Age']
    
    return initial_df
    
X_train_addedColumns = add_features(X_train_addedColumns)
X_test_addedColumns = add_features(X_test_addedColumns)
```


```python
# Standardize

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_addedColumns[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', \
                    'DiabetesPedigreeFunction', 'Age', 'Glucose_squared', 'Glucose_cubed', 'BMI_squared', \
                    'BMI_cubed', 'DiabetesPedigreeFunction_squared', 'DiabetesPedigreeFunction_cubed', \
                    'Pregnancies_x_Age', 'BP_x_Glucose', 'DPF_x_Glucose', 'BMI_x_Age']] = \
    scaler.fit_transform(X_train_addedColumns[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', \
                    'DiabetesPedigreeFunction', 'Age', 'Glucose_squared', 'Glucose_cubed', 'BMI_squared', \
                    'BMI_cubed', 'DiabetesPedigreeFunction_squared', 'DiabetesPedigreeFunction_cubed', \
                    'Pregnancies_x_Age', 'BP_x_Glucose', 'DPF_x_Glucose', 'BMI_x_Age']])
    
X_test_addedColumns[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', \
                    'DiabetesPedigreeFunction', 'Age', 'Glucose_squared', 'Glucose_cubed', 'BMI_squared', \
                    'BMI_cubed', 'DiabetesPedigreeFunction_squared', 'DiabetesPedigreeFunction_cubed', \
                    'Pregnancies_x_Age', 'BP_x_Glucose', 'DPF_x_Glucose', 'BMI_x_Age']] = \
    scaler.fit_transform(X_test_addedColumns[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', \
                    'DiabetesPedigreeFunction', 'Age', 'Glucose_squared', 'Glucose_cubed', 'BMI_squared', \
                    'BMI_cubed', 'DiabetesPedigreeFunction_squared', 'DiabetesPedigreeFunction_cubed', \
                    'Pregnancies_x_Age', 'BP_x_Glucose', 'DPF_x_Glucose', 'BMI_x_Age']])
```


```python
# Encoding

X_train_encoded = pd.get_dummies(X_train_addedColumns)
X_test_encoded = pd.get_dummies(X_test_addedColumns)
```


```python
print("Cross validation:")
for i, model in enumerate(models):
    scores = cross_val_score(model, X_train_encoded, y_train, cv=k_folds, scoring='accuracy')
    print(model_names[i], ": ", scores.mean())
```

    Cross validation:
    Logistic Regression :  0.740977372492
    Naive Bayes :  0.653167765152
    Random Forest :  0.750825906197
    Extra Trees :  0.747481007597
    XGBoost :  0.749147007864
    

### Feature Selection


```python
# feature importance
from xgboost import plot_importance
import eli5
from eli5.sklearn import PermutationImportance

model_importance = XGBClassifier()

model_importance.fit(X_train_encoded, y_train)

#perm = PermutationImportance(model_importance, cv=k_folds)
#perm.fit(X_train_encoded, y_train)
#eli5.show_weights(perm)

#plot_importance(model_importance)
#plt.show()
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=1, verbosity=1)




```python
feature_scores = pd.DataFrame({'Features': X_train_encoded.columns, 'F Score': model_importance.feature_importances_})

sorted_features = feature_scores.sort_values('F Score', ascending = False)

top_features = list(sorted_features.Features[0:6])

sorted_features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F Score</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.216864</td>
      <td>Glucose</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.146590</td>
      <td>BMI_x_Age</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.081636</td>
      <td>BMI</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.076336</td>
      <td>DPF_x_Glucose</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.075118</td>
      <td>Insulin</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train_selected = X_train_encoded[top_features]
X_test_selected = X_test_encoded[top_features]
```


```python
# Correlation Matrix

plt.figure(figsize=(10, 8))
sns.heatmap(X_train_selected.corr(), annot=True)
plt.show()
```


#![png](output_32_0.png)



```python
print("Cross validation:")
for i, model in enumerate(models):
    scores = cross_val_score(model, X_train_selected, y_train, cv=k_folds, scoring='accuracy')
    print(model_names[i], ": ", scores.mean())
```

    Cross validation:
    Logistic Regression :  0.770325848155
    Naive Bayes :  0.752518777435
    Random Forest :  0.770285649181
    Extra Trees :  0.76871294493
    XGBoost :  0.768686933829
    

### Hyperparameter Tuning


```python
### Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

# Number of trees
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 30)]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10, num = 8)]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Minimum loss reduction per node
gamma = [0, 0.5, 1, 1.5, 2, 5]


# Grid Parameters
rf_random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

xgb_random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'gamma': gamma}

```


```python
# Grid Search for Random Forest
rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), \
                               param_distributions = rf_random_grid, \
                               n_iter = 200, \
                               cv = k_folds)

rf_random.fit(X_train_selected, y_train)

rf_best_random = rf_random.best_estimator_
```


```python
# Grid Search for Extra Trees
et_random = RandomizedSearchCV(estimator = ExtraTreesClassifier(), \
                               param_distributions = rf_random_grid, \
                               n_iter = 200, \
                               cv = k_folds)

et_random.fit(X_train_selected, y_train)

et_best_random = et_random.best_estimator_
```


```python
# Grid Search for XGBoost

xgb_random = RandomizedSearchCV(estimator = XGBClassifier(), \
                               param_distributions = xgb_random_grid, \
                               n_iter = 200, \
                               cv = k_folds)

xgb_random.fit(X_train_selected, y_train)

xgb_best_random = xgb_random.best_estimator_
```


```python
models_best = [LogisticRegression(solver = 'liblinear'), GaussianNB(), \
          rf_best_random, et_best_random, xgb_best_random]
```


```python
print("Cross validation:")
for i, model in enumerate(models_best):
    scores = cross_val_score(model, X_train_selected, y_train, cv=k_folds, scoring='accuracy')
    print(model_names[i], ": ", scores.mean())
```

    Cross validation:
    Logistic Regression :  0.770325848155
    Naive Bayes :  0.752518777435
    Random Forest :  0.77842948842
    Extra Trees :  0.771964762482
    XGBoost :  0.781775031923
    


```python
print("Test Set:")
for i, model in enumerate(models_best):
    predictions = model.fit(X_train_selected, y_train).predict(X_test_selected)
    accuracy = metrics.accuracy_score(y_test, predictions)
    print(model_names[i], ": ", accuracy)
```

    Test Set:
    Logistic Regression :  0.766233766234
    Naive Bayes :  0.779220779221
    Random Forest :  0.798701298701
    Extra Trees :  0.779220779221
    XGBoost :  0.818181818182
    

### Stacking Models


```python
from scipy import stats

# Model Ensemble

model_predict = np.zeros((len(X_train_selected), len(models_best)))

for i, model in enumerate(models):
    
    model_predict[:, i] = cross_val_predict(model, X_train_selected, y_train, cv=k_folds)
    
    #model_predict[:,i] = model.fit(X_train, y_train).predict(X_test)
    
majority_vote = stats.mode(model_predict, axis = 1)[0]
print(metrics.accuracy_score(y_train, majority_vote))
```

    0.770358306189
    

### Run Test Set for Final Prediction


```python
# Final XGB model for predictions 

final_predictions = xgb_best_random.fit(X_train_selected, y_train).predict(X_test_selected)
    
print("Accuracy: ", round(metrics.accuracy_score(y_test, final_predictions) * 100, 1), "%")
```

    Accuracy:  81.8 %
    


```python

```
