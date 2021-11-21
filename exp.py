import shap
import pickle as pkl
import pandas as pd

# loading data

test = pd.read_csv(r'test_data.csv')

# loading the models

with open('xgboost/model.pkl', 'rb') as f:
                model = pkl.load(f)
print(test.shape)
background = shap.maskers.Independent(test[:1000])
explainer = shap.Explainer(model.predict, background, link=shap.links.logit)
shap_values = explainer(test[:1000])


# explain the model's predictions using SHAP values
background = shap.maskers.Independent(test[:1000])
def f(x):
    return shap.links.identity(model.predict_proba(x, validate_features=False)[:,1])
explainer = shap.Explainer(f, background, link=shap.links.logit)
shap_values = explainer(test[:1000])

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[2])


shap.plots.initjs()

# visualize the first prediction's explanation
shap.plots.force(shap_values[0])


# plot the global importance of each feature
shap.plots.bar(shap_values[0])

# visualize the first prediction's explanation
shap.plots.force(shap_values)

# plot the importance of a single feature across all samples
shap.summary_plot(shap_values)

# plot the global importance of each feature
shap.plots.bar(shap_values)

# plot the distribution of importances for each feature over all samples
shap.plots.beeswarm(shap_values)