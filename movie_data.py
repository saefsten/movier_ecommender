import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
from joblib import load
from difflib import get_close_matches

class MovieData:

    def __init__(self):
        '''
        Gets the movies and ratings from the database. Clean their names and creates one dict for mapping title>>id and one id<<title.
        '''
        self.ratings = pd.read_csv('ml-latest-small/ratings.csv', sep=',')
        self.movie_average = (self.ratings.groupby('movieId')['rating'].mean()).to_dict()
        self.ratings.drop(['timestamp'], axis=1, inplace=True)
        self.ratings.set_index('userId', inplace=True)
        self.R = self.ratings.pivot(index=self.ratings.index, columns='movieId')['rating']
        self.R.fillna(value=self.movie_average, inplace=True)
  
        '''create mapping dictionary'''
        self.movies = pd.read_csv('ml-latest-small/movies.csv', sep=',')
        self.movies['title_only'] = self.movies['title'].str.extract(r'(.+?(?=\s\(\d\d\d\d\)))', expand=True)
        self.movies['title_only'] = self.movies['title_only'].astype('string')
        self.movies['title_only'].fillna(value='xxx', inplace=True)
        self.movies['title_only'].apply(lambda x: x.strip())
        self.movieid_dict = self.movies.set_index('title_only').to_dict()['movieId'] # to get movieId from input title
        self.movie_dict = self.movies.set_index('movieId').to_dict()['title'] # to convert back from movieId to title
        self.movies_list = self.movies['title_only'].tolist()

    def movie_available(self, movies):
        '''
        Accepts a list of movie titels and check if they are in the database.
        If not, a seach for similar titles are done and the most similar title is returned together with the known titles.
        '''
        self.movies_to_check = movies
        self.check = []
        self.score = 0
        self.suggested_movies = []
        for movie in self.movies_to_check:
            # if self.movies['title_only'].str.contains(movie).any(): # it finds incorrect macthes, ie "Titani" is in "Titanic", but then "Titani" is not found on line 64
            if str(movie) in self.movies_list:
                self.check.append(("{} is available").format(movie))
                self.suggested_movies.append(movie)
            else:
                if len((get_close_matches(movie, self.movies_list))) > 0:
                    suggestion = get_close_matches(movie, self.movies_list)[0]
                    self.score += 1
                    self.check.append(("Movie '{}' is not in our database. Did you mean {}?").format(movie, suggestion))
                    self.suggested_movies.append(suggestion)
                else:
                    self.check.append(("Movie '{}' is not in the our database").format(movie))
                    self.score += 1
        return self.check, self.suggested_movies, self.score

    def get_recommendation(self, movies, ratings):
        """ input movies and ratings
            process data and get nmf model
            use mask to avoid recommending the same movies as in input
            return a list witht he top three recommended movies
        """
        self.movies = movies
        self.ratings = ratings
        
        """replace movie title with column and create new dict"""
        self.movieids = [self.movieid_dict[self.movies[0]], self.movieid_dict[self.movies[1]], self.movieid_dict[self.movies[2]]]
        self.ratings = list(map(int, self.ratings)) # convert ratings to integers
        self.user_input = dict(zip(self.movieids, self.ratings))

        '''create df with user_input and fill na'''
        self.new_user = pd.DataFrame(data=self.user_input, index=['user'], columns=self.R.columns)
        self.mask = [0 if x > 0 else 1 for x in self.new_user.loc['user']]
        self.new_user.fillna(value=self.movie_average, inplace=True)

        '''load model'''
        self.nmf = load('nmf.joblib')
        self.Q = self.nmf.components_

        '''make predicitons'''
        self.P_user = self.nmf.transform(self.new_user)
        self.R_user = pd.DataFrame(np.dot(self.P_user, self.Q), index=['user'], columns=self.R.columns)
        self.filter_user = self.R_user.loc['user'] * self.mask
        self.filter_user.sort_values(ascending=False)
        self.user_recommendation = self.filter_user.sort_values(ascending=False)

        """final list and replace movies column with movie title"""
        self.user_recommendation_list = [self.user_recommendation.index[0], self.user_recommendation.index[1], self.user_recommendation.index[2]]
        self.user_recommended_movies = [self.movie_dict[self.user_recommendation_list[0]], self.movie_dict[self.user_recommendation_list[1]], self.movie_dict[self.user_recommendation_list[2]]]
        return self.user_recommended_movies