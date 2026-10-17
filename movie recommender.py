import pandas as pd
from sklearn.neighbors import NearestNeighbors

# movies with action and romance scores out of 10
data = {
    "movie": ["Speed", "Die Hard", "Titanic", "The Notebook", "Avengers", "La La Land"],
    "action": [9, 10, 2, 1, 9, 1],
    "romance": [2, 1, 10, 9, 3, 8]
}

df = pd.DataFrame(data)

# features: action and romance scores
x = df[["action", "romance"]]

# find the 2 nearest movies
model = NearestNeighbors(n_neighbors=2)
model.fit(x)

# your taste: action score 8, romance score 2
your_taste = pd.DataFrame([[8, 2]], columns=["action", "romance"])

# find closest movies to your taste
distances, indices = model.kneighbors(your_taste)

print("movies recommended for you:")
for idx in indices[0]:
    print("-", df.iloc[idx]["movie"])
