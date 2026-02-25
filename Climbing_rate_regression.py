################################
####### Import libraries #######
################################

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


##############################################
####### Import, clean, preprocess data #######
##############################################

# Mountain Project datafiles (116k).
mp = pd.read_csv("archive/mp_routes.csv", index_col=0)
mp.columns = mp.columns.str.strip()
print(f"Full dataset: {len(mp)} routes")

# Clean and preprocess data.
mp_clean = mp[
    (mp["Pitches"] == 1) &                 # Single pitch routes only
    (mp["Route Type"] == "Sport") &        # Sport climbing only
    (mp["num_votes"] >= 5) &               # Minimum 5 votes for reliability
    (mp["desc"].notna())                   # Ensure description is not missing  
].copy()

# Ensure stability.
mp_clean["log_votes"] = np.log1p(mp_clean["num_votes"])

# Convert YDS climbing ratings to numeric scale.
def yds_to_num(r):
    m = re.search(r"5\.(\d+)([abcd]?)([+-]?)", str(r))
    if not m:
        return np.nan
    base = int(m.group(1))
    letter = {"a":0, "b":0.25, "c":0.5, "d":0.75}.get(m.group(2), 0)
    modifier = {"+":0.2, "-":-0.2}.get(m.group(3), 0)
    return base + letter + modifier

mp_clean["grade_num"] = mp_clean["Rating"].apply(yds_to_num).replace(0, np.nan)
# Remove climbs w/o grading information. 
mp_clean = mp_clean.dropna(subset=["grade_num"]).copy()
# Select climbs above climbing rate 5.8
mp_clean = mp_clean.loc[(mp_clean["grade_num"]>=8)].copy()

print(f"Filtered dataset: {len(mp_clean)} routes")


#################################################
####### Text feature engineering (TF-IDF) #######
#################################################

# Vectorizing is for testing/engineering
vectorizer_test = TfidfVectorizer(
    max_features=5000,
    min_df=5,                 # Keep words that appear in at least 5 routes
    stop_words="english",     # English stopwords
    ngram_range=(1,1),        # Unigrams only
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # only alphabetic words len>=2
)
X_text_test = vectorizer_test.fit_transform(mp_clean["desc"]) # learn IDF and produce TF-IDF matrix
feature_names_test = vectorizer_test.get_feature_names_out()

# Print top 20 most common words in descriptions to engineer features.
word_counts = np.asarray(X_text_test.sum(axis=0)).flatten()
top_idx = np.argsort(word_counts)[::-1][:20]
print("Top 20 most common words and count occurence in route descriptions:")
for word, count in zip(feature_names_test[top_idx], word_counts[top_idx]):
    print(word, int(count))

# Remove words that are common in climbing but not informative for predicting ratings.
custom_stop = {"way","past","just","good","great","fun","start","anchors","wall","rock","line","left","right","climb", "climbing","route","feet","ft","meter","meters","pitch","pitches"}

vectorizer = TfidfVectorizer(
    stop_words=list(set(TfidfVectorizer(stop_words="english").get_stop_words()) | custom_stop),
    ngram_range=(1,2),       # Unigrams and bigrams
    min_df=20,
    max_features=5000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X_text = vectorizer.fit_transform(mp_clean["desc"])
feature_names = vectorizer.get_feature_names_out()



#################################
####### Create dictionary #######
#################################

# Used full output list of words (5k) as input for chatGPT to create a dictionary of climbing features.

angle_dict = {
    "slab": [
        "slab", "slabby", "slabbing", "low angle", "low angled",
        "slab face", "vertical slab", "technical slab", "slab crux"
    ],
    "vertical": [
        "vertical", "vertical face", "vertical section",
        "vertical terrain", "dead vertical"
    ],
    "overhang": [
        "overhang", "overhanging", "overhung",
        "steep", "steeper", "steep face",
        "steep section", "steep roof", "overhanging face",
        "overhanging section"
    ],
    "roof": [
        "roof", "roofs", "roof crux", "roof pull",
        "pull roof", "roof section", "lip", "ceiling"
    ]
}

feature_dict = {
    "crack": [
        "crack", "cracks", "hand crack", "finger crack",
        "wide crack", "offwidth", "crack corner"
    ],
    "arete": [
        "arete", "aretes", "arete crux", "rounded arete"
    ],
    "dihedral": [
        "corner", "corners", "dihedral", "dihedrals",
        "shallow dihedral", "facing dihedral"
    ],
    "flake": [
        "flake", "flakes", "large flake", "detached flake"
    ],
    "chimney": ["chimney"]
}

hold_dict = {
    "crimpy": [
        "crimp", "crimps", "crimpy", "small crimps",
        "tiny crimps", "sharp crimps"
    ],
    "juggy": [
        "jug", "jugs", "juggy", "big jugs",
        "huge jugs", "bucket", "buckets"
    ],
    "pockets": [
        "pocket", "pockets", "pocketed",
        "finger pocket", "mono"
    ],
    "slopers": [
        "sloper", "slopers", "slopey",
        "sloping", "sloping holds"
    ],
    "edges": [
        "edge", "edges", "small edges",
        "positive edges"
    ],
    "undercling": [
        "undercling", "underclings"
    ]
}

movement_dict = {
    "technical": [
        "technical", "techy", "delicate",
        "precise", "balance", "balancy",
        "footwork"
    ],
    "powerful": [
        "powerful", "burly", "bouldery",
        "dynamic", "athletic"
    ],
    "pumpy": [
        "pumpy", "endurance", "sustained",
        "power endurance", "fight pump"
    ],
    "reachy": [
        "reachy", "long reach", "big reach",
        "long reaches"
    ],
    "runout": [
        "runout", "runouts", "spaced bolts"
    ]
}

themes = {
    "angle": angle_dict,
    "feature": feature_dict,
    "hold": hold_dict,
    "movement": movement_dict
}






#################################
####### Linear regression #######
#################################


# Make everything lower case.
desc = mp_clean["desc"].str.lower()

# Binary flags from dictionaries to prepare for linear regression
style_cols = []
for theme in themes.values():
    for style, words in theme.items():
        mp_clean[style] = desc.str.contains("|".join(map(lambda w: rf"\b{w}\b", words)), regex=True).astype(int)
        style_cols.append(style)

# Keep only rows with outcome + predictors present
mp_clean = mp_clean.dropna(subset=["Avg Stars", "grade_num", "log_votes"]).copy()

# Fit linear regression using OLS. 
X = sm.add_constant(mp_clean[["grade_num", "log_votes"] + style_cols])
y = mp_clean["Avg Stars"]

m = sm.OLS(y, X).fit(cov_type="HC3") 

# Print table of "what drives stars" 
res = (pd.DataFrame({
        "coef": m.params,
        "se": m.bse,
        "t": m.tvalues,
        "p": m.pvalues
     })
     .loc[style_cols]
     .assign(abs_coef=lambda d: d["coef"].abs())
     .sort_values(["abs_coef", "p"], ascending=[False, True])
)
print(m.summary())
res.head(20)




###########################
####### Create plot #######
###########################

plot_df = res.copy()

# 95% CI
ci = m.conf_int().loc[style_cols]
plot_df["ci_low"]  = ci[0]
plot_df["ci_high"] = ci[1]

# Sort for nice plotting
plot_df = plot_df.sort_values("coef")

plt.figure(figsize=(8,10))

plt.errorbar(
    plot_df["coef"],
    plot_df.index,
    xerr=[plot_df["coef"] - plot_df["ci_low"],
          plot_df["ci_high"] - plot_df["coef"]],
    fmt='*', ms=14,
    color='mediumorchid',
    ecolor='cornflowerblue',
    elinewidth=4,
    capsize=6)

plt.xlim(-0.1, 0.22)
plt.axvline(0, linestyle='--',color='grey')
plt.xlabel(r"Regression coefficient $\beta_k$ ")
plt.title("Which features drive star ratings?")
plt.tight_layout()
plt.savefig("Stars_Regression_Coefficient_Grading_above8.png", dpi=300)
plt.show()




#############################
####### Sanity checks #######
#############################

# Check average stars that are or are not pumpy. 
pumpy_means = mp_clean.groupby("pumpy")["Avg Stars"].mean()


# Check multicollinearity
X_styles = mp_clean[style_cols] # Only style columns

vif = pd.DataFrame()
vif["feature"] = style_cols
vif["VIF"] = [
    variance_inflation_factor(X_styles.values, i)
    for i in range(len(style_cols))
]

vif.sort_values("VIF", ascending=False)
