# %% [markdown]
# # Made by Muhammad Dava Pasha
# # Dicoding Username: mdavap

# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ## Import Data

# %%
df = pd.read_csv("./winequality-red.csv")

# %%
df.info()

# %%
df.head()

# %% [markdown]
# ## Check missing values

# %%
df.isna().sum()

# %%
df.isnull().sum()

# %% [markdown]
# ## Explore the data

# %%
sns.pairplot(df, diag_kind='kde', hue='quality')

# %%
df.describe().transpose()

# %%
plt.figure(figsize = (16,16))
corr = df.corr()
ax = sns.heatmap(corr, cmap="YlGnBu", annot=True, mask=np.triu(corr))

# %% [markdown]
# ### Check outlier

# %%
columns = df.columns.values[:-1] # Exclude quality
columns_size = columns.size
fig, ax = plt.subplots(columns_size, 1, figsize=(15, 45)) 
for i in range(columns_size):
    sns.boxplot(x=columns[i], data=df, ax=ax[i])

# %% [markdown]
# ### Result from boxplot is that we can see there no outlier

# %% [markdown]
# ## Modeling Machine Learning using Random Forest Regressor

# %% [markdown]
# ### Splitting the data for training and testing

# %%
X = df[columns]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

# %% [markdown]
# ### Model development and prediction

# %%
regressor = RandomForestRegressor(random_state = 100)  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test) 

# %% [markdown]
# ### Model Evaluation

# %%
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# %%
cv_scores = cross_val_score(regressor, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f}")

# %% [markdown]
# ### Model improvement using GridSearchCV

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=45),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
the_best_model = grid_search.best_estimator_

# %%
print("The best parameter from grid search: ")
grid_search.best_params_

# %%
best_predictions = the_best_model.predict(X_test)
print("Best Model Performance:")
print(f"R² Score: {r2_score(y_test, best_predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_predictions)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, best_predictions):.4f}")

# %%
cv_scores = cross_val_score(the_best_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f}")

# %% [markdown]
# ## Modeling Machine Learning using Linear Regression

# %% [markdown]
# ### Model development and prediction

# %%
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test) 

# %% [markdown]
# ### Model Evaluation

# %%
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")

rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")

# %%
cv_scores = cross_val_score(regressor, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f}")


