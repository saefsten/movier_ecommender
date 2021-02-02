from flask import Flask, render_template, request
from movie_data import MovieData
import logging
# FLASK_APP=application.py flask run
# logging.basicConfig(filename='log_application.log')
# __name__ is a reference to the current script (application.py)

app = Flask(__name__) # instantiating a Flask application

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    user_input = dict(request.args)
    user = user_input['user']
    movies = [user_input['movie1'], user_input['movie2'], user_input['movie3']]
    ratings = [user_input['rating1'], user_input['rating2'], user_input['rating3']]

    if (movies[0]=='') or (movies[1]=='') or (movies[2]==''):
        return render_template('missing.html', movies=movies, ratings=ratings, user=user)

    moviedata = MovieData()
    
    check, suggestions, score = moviedata.movie_available(movies)
    print(check, suggestions, score)
    if score != 0:
        results = check
        return render_template('failed.html', results=results, suggestions=suggestions, ratings=ratings, user=user)
    else:
        results = moviedata.get_recommendation(movies, ratings)
        return render_template('recommendations.html', results=results, user=user)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
