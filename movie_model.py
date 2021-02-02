import pandas as pd
from sklearn.decomposition import NMF
from joblib import dump
# from sqlalchemy import create_engine
import os
from tqdm import tqdm

# password = os.getenv('POSTGRES_PASSWORD')
# user = os.getenv('POSTGRES_USER')

# pg = create_engine('postgres://postgres:postgres1@localhost:5432/movies') #.format(user,password))

# create_query = """
# CREATE TABLE IF NOT EXISTS tweets (
#     id INTEGER,
#     text TEXT,
#     sentiment FLOAT,
#     status INTEGER
# );
# """
# pg.execute(create_query)

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movie_average = (ratings.groupby('movieId')['rating'].mean().round(1)).to_dict()
ratings.drop(['timestamp'], axis=1, inplace=True)
ratings.set_index('userId', inplace=True)
R = ratings.pivot(index=ratings.index, columns='movieId')['rating']
R.fillna(value=movie_average, inplace=True)

# R.iloc[0:100,:].to_csv('R1.csv')
# R.iloc[100:200,:].to_csv('R2.csv')
# R.iloc[200:300,:].to_csv('R3.csv')
# R.iloc[300:400,:].to_csv('R4.csv')
# R.iloc[400:500,:].to_csv('R5.csv')
# R.iloc[500:,:].to_csv('R6.csv')
# R.to_sql('movies', pg)

nmf = NMF(n_components=25, max_iter=10000)
tqdm(nmf.fit(R))
dump(nmf, 'nmf.joblib')