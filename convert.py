import joblib 

model_path = 'xgb_model_5GNIDD.pkl'
model_path_2 = 'xgb_model_MSA.pkl'

m = joblib.load(model_path)
n = joblib.load(model_path_2)

# Converting the model to json (needed to compute all minimal using vote-xai)
import vote
vote_explainer = vote.Ensemble.from_xgboost(m)
vote_explainer_2 = vote.Ensemble.from_xgboost(n)


with open('5GNIDD.json', 'w') as f:
    f.write(vote_explainer.serialize())

with open('MSA.json', 'w') as f:
    f.write(vote_explainer_2.serialize())

