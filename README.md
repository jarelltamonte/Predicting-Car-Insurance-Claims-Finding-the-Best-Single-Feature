# Predicting Car Insurance Claims: Finding the Best Single Feature

![Car](car.jpg)

## Overview

Insurance companies invest a lot of [time and money](https://www.accenture.com/_acnmedia/pdf-84/accenture-machine-leaning-insurance.pdf) into optimizing their pricing and accurately estimating the likelihood that customers will make a claim. In many countries, car insurance is a legal requirement for driving on public roads, so the market is very large.

This project is a **simulated scenario**: I imagined working with a car insurance company, "On the Road," that wants a model to predict whether a customer will make a claim during the policy period. With very little expertise and infrastructure for deploying and monitoring machine learning models, the company would want to **start with a simple model in production**. So the task is to identify the **single feature that gives the best-performing model, measured by accuracy**. The company is fictional and no real request was made.

## Goal

Build a separate logistic regression model for each feature in the dataset and find the one that predicts the `outcome` (claim or no claim) with the highest accuracy.

## Dataset

`car_insurance.csv` contains customer data for 10,000 clients.

| Column | Description |
|--------|-------------|
| `id` | Unique client identifier |
| `age` | Client's age group: `0`: 16-25, `1`: 26-39, `2`: 40-64, `3`: 65+ |
| `gender` | `0`: Female, `1`: Male |
| `driving_experience` | Years the client has been driving: `0`: 0-9, `1`: 10-19, `2`: 20-29, `3`: 30+ |
| `education` | `0`: No education, `1`: High school, `2`: University |
| `income` | `0`: Poverty, `1`: Working class, `2`: Middle class, `3`: Upper class |
| `credit_score` | Client's credit score (between zero and one) |
| `vehicle_ownership` | `0`: Does not own their vehicle (paying off finance), `1`: Owns their vehicle |
| `vehicle_year` | Year of vehicle registration: `0`: Before 2015, `1`: 2015 or later |
| `married` | `0`: Not married, `1`: Married |
| `children` | Client's number of children |
| `postal_code` | Client's postal code |
| `annual_mileage` | Number of miles driven by the client each year |
| `vehicle_type` | `0`: Sedan, `1`: Sports car |
| `speeding_violations` | Total number of speeding violations received by the client |
| `duis` | Number of times the client has been caught driving under the influence of alcohol |
| `past_accidents` | Total number of previous accidents the client has been involved in |
| `outcome` | **Target variable.** `0`: No claim, `1`: Made a claim |

## Approach

1. **Load the data** with pandas.
2. **Handle missing values** by filling the missing `credit_score` and `annual_mileage` values with each column's mean. After this, all 10,000 rows are complete.
3. **Build one model per feature:** for every column except `id` and `outcome`, fit a logistic regression with `statsmodels` (`outcome ~ feature`).
4. **Compute accuracy** for each model from its confusion matrix: (true negatives + true positives) divided by all predictions.
5. **Pick the best feature** as the one with the highest accuracy and store it in a DataFrame.

## Results

| Best feature | Accuracy |
|--------------|----------|
| `driving_experience` | **0.7771** (about 77.7%) |

**Driving experience** is the single best predictor of whether a customer will make a claim, with an accuracy of roughly **77.7%**.

### Key Takeaways

- If the company wants to start with a simple one-feature model, **driving experience** is the best choice among the features tested.
- A single feature reaching about 78% accuracy is a strong starting point that is easy to deploy and monitor.
- Accuracy alone can be misleading if claims and non-claims are not evenly balanced, so metrics like precision and recall are worth checking before going to production.

## Tools and Libraries

- Python
- pandas
- NumPy
- statsmodels (`logit`)
- Jupyter Notebook

## Repository Structure

```
├── README.md
├── notebook.ipynb         # Analysis code
├── car.jpg                # README header image
└── car_insurance.csv      # Dataset
```

## Possible Next Steps

- Evaluate on a held-out test set. The accuracies here are measured on the same data the models were trained on.
- Compare accuracy with precision, recall, and F1, especially if the classes are imbalanced.
- Combine several features into one model and compare it against the best single feature.
- Try other models, such as decision trees or random forests, for a more flexible approach.
