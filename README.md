# Linear Regression from Scratch – Boston Housing Dataset

##  Project Overview

This project demonstrates the implementation of **Linear Regression from scratch** on the **Boston Housing Dataset**, with model evaluation performed using **sklearn’s regression metrics**. The aim is to understand the internal working of linear regression while leveraging reliable library-based metrics for evaluation.

---

##  Project Objective

* Implement linear regression logic manually (weights and bias)
* Use the learned parameters to generate predictions
* Evaluate model performance using **sklearn metrics**
* Understand how model performance changes with limited features

---

##  Dataset Used

**Boston Housing Dataset**

* **Target Variable:**

  * `MEDV` – Median value of owner-occupied homes (in $1000s)

* **Features (example subset):**

  * `RM` – Average number of rooms per dwelling
  * `DIS` – Distance to employment centers
  * `ZN` – Proportion of residential land zoned
  * `CHAS` – Charles River dummy variable (0 or 1)
  * `B` – Proportion of Black population (historical variable)

Depending on the experiment, single-feature and multi-feature regression were explored.

---

##  Model Description

For **single-feature linear regression**, the model follows:

[ \hat{y} = wx + b ]

Where:

* `w` is the learned weight
* `b` is the bias/intercept
* `x` is the input feature
* `ŷ` is the predicted output

The parameters are learned using **Gradient Descent** implemented from scratch.

---

##  Prediction Logic (From Scratch)

```python
def predict(X, w, b):
    return w * X + b
```

---

##  Model Evaluation (Using sklearn)

Since this is a **regression problem**, accuracy is not applicable. The following **sklearn metrics** are used:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

###  Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

```python
mse = mean_squared_error(y_true, y_pred)
```

---

###  Root Mean Squared Error (RMSE)

Provides error in the same units as the target variable.

```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

---

###  Mean Absolute Error (MAE)

Measures average absolute prediction error.

```python
mae = mean_absolute_error(y_true, y_pred)
```

---

###  R² Score

Indicates the proportion of variance in the target variable explained by the model.

```python
r2 = r2_score(y_true, y_pred)
```

---

## 📈 Results Summary

| Metric                  | Value |
| ----------------------- | ----- |
| R² Score                | ~0.48 |
| Mean Squared Error      | ~39   |
| Root Mean Squared Error | ~6.2  |

### Interpretation:

* The model explains approximately **48% of the variance** in house prices
* Average prediction error is around **$6,000**
* Results are expected given limited feature usage

---

## Notes & Limitations

* Model performance is constrained due to limited features
* No regularization applied
* Linear regression is sensitive to outliers

These factors contribute to a moderate R² score.

---

##  Key Learnings

* Practical understanding of linear regression internals
* Correct usage of sklearn regression metrics
* Handling shape-related issues (1D vs 2D arrays)
* Interpreting regression results realistically

---

##  Future Scope

* Extend to full multiple linear regression
* Apply feature scaling
* Compare scratch implementation with sklearn’s LinearRegression
* Improve performance using better feature selection

---

##  Tech Stack

* Python
* NumPy
* scikit-learn

---

##  Author

**Bharvesh Dabas**

---

##  Note

This project is focused on **conceptual clarity and interview readiness**, emphasizing understanding over blind library usage.
