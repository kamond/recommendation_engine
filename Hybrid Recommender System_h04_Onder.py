import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)


"""
df['userId'].sample(n=3)

df_users_test = df.groupby('userId').count()

df_users_test.sort_values('title', ascending=False)
"""

################################################################################################
# Görev1: Veri Ön İşleme İşlemlerini Gerçekleştiriniz
################################################################################################

def create_user_movie_df():
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df



user_movie_df = create_user_movie_df()


user_movie_df.shape



################################################################################################
# Görev2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
################################################################################################


random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

random_user

random_user_df = user_movie_df[user_movie_df.index == random_user]

random_user_df.head()

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

movies_watched

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Home Alone (1990)"]

len(movies_watched)

################################################################################################
# Görev3: Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz..
################################################################################################


movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

movies_watched_df.shape


user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count.head()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

users_same_movies.head()
users_same_movies.count()
users_same_movies.index

################################################################################################
# Görev4: Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz
################################################################################################

movies_watched_df[movies_watched_df.index.isin(users_same_movies)].shape

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df.shape
final_df.index.nlevels
final_df.head()
final_df.T
final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.nlevels

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df.head()

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head(3)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

################################################################################################
# Görev5: Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
################################################################################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.head()

top_users_ratings.shape

top_users_ratings["movieId"].nunique()

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})


recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})


recommendation_df.shape
recommendation_df.index.nlevels

recommendation_df = recommendation_df.reset_index()

recommendation_df[["movieId"]].nunique()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

movies_to_be_recommend_merged = movies_to_be_recommend.merge(movie[["movieId", "title"]])


#ilk 5 film:
top5_weighted_avg = movies_to_be_recommend_merged.iloc[0:5]

top5_weighted_avg

################################################################################################
# Görev6: Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
################################################################################################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

user = 108170

movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
    sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_name = movie[movie["movieId"] == movie_id]

# User Based Recommendation
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].iloc[0:5]

# Item Based Recommendation
item_based_top_five = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(5)

item_based_top_five.index