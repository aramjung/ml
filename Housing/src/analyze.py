import pandas as pd 
import transform as tranform 
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv("../data/train.csv")
y = df.pop('SalePrice')

categorical_features = df.select_dtypes(include=['object']).columns
all_features = df.columns 

df = tranform.encode_fit_transform(df, y, categorical_features, True)

# Mutual information analysis
mi_scores = mutual_info_regression(df, y)
mi_scores = pd.Series(mi_scores, index=all_features).sort_values(ascending=False)

mi_scores_df = mi_scores.reset_index()
mi_scores_df.columns = ["Feature", "MI_Score"]
pd.set_option('display.max_rows', None)
print(mi_scores_df)

low_mi_features = mi_scores_df[mi_scores_df["MI_Score"] <= 0.01].iloc[:,0]
# compute mi score for all pairs of weak features
interaction_mi_scores = {}
for i in range(len(low_mi_features)):
    for j in range(i+1, len(low_mi_features)):
        feature1 = low_mi_features.iloc[i]
        feature2 = low_mi_features.iloc[j]
        new_feature = df[feature1] * df[feature2]
        mi_score = mutual_info_regression(pd.DataFrame(new_feature), y)
        interaction_mi_scores[(feature1, feature2)] = mi_score

interaction_mi_df = pd.DataFrame(interaction_mi_scores.items(), columns=["Feature Pair", "MI_Score"])
interaction_mi_df = interaction_mi_df.sort_values(by="MI_Score", ascending=False)

# Display the top 10 feature interactions
print(interaction_mi_df[interaction_mi_df['MI_Score'] > 0.01])



