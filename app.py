from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
import numpy as np
import gdown
import os
from collections import Counter

app = Flask(__name__)

# Google Drive File IDs (Replace with your actual file IDs)
file_links = {
    "similarity_tfidf.pkl": "https://drive.google.com/file/d/1Oo4E8oM2W16RItwNxCdrIm7-rpdZ2ka7/view?usp=sharing",
    "similarity_lsi.pkl": "https://drive.google.com/file/d/1ruTpXSE68oXzt9_gSHOrb2VMjDKxoNMU/view?usp=drive_link",
    "similarity_bm25.pkl": "https://drive.google.com/file/d/1ks0MU0JWZG3gr6eCUEnJOzmLGEyBEwer/view?usp=drive_link",
    "similarity_word2vec.pkl": "https://drive.google.com/file/d/1iFXseQIArJeyQ80i44DMbAR697PKrvNP/view?usp=sharing",
    "similarity_jaccard.pkl": "https://drive.google.com/file/d/1hPoAq9OAWe9d1JipcM0g2addssgADCF4/view?usp=drive_link"
}

# Function to download files from Google Drive
def download_files():
    for filename, file_id in file_links.items():
        if not os.path.exists(filename):  # Check if file already exists
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename} from Google Drive...")
            gdown.download(url, filename, quiet=False)

# Download the .pkl files if not present
download_files()

# Load preprocessed data
movies = pd.read_csv('preprocessed_movies.csv')
movies['title'] = movies['title'].str.lower()

# Load similarity models
similarity_tfidf = np.array(pickle.load(open('similarity_tfidf.pkl', 'rb')))
similarity_lsi = np.array(pickle.load(open('similarity_lsi.pkl', 'rb')))
similarity_bm25 = np.array(pickle.load(open('similarity_bm25.pkl', 'rb')))
similarity_word2vec = np.array(pickle.load(open('similarity_word2vec.pkl', 'rb')))
similarity_jaccard = np.array(pickle.load(open('similarity_jaccard.pkl', 'rb')))

# TMDb API Key
API_KEY = "9bf054e3ab828da7dd1c53fec9e3f009"
poster_cache = {}

def fetch_movie_poster(movie_name):
    """Fetch movie poster URL from TMDb API."""
    if movie_name in poster_cache:
        return poster_cache[movie_name]

    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    response = requests.get(search_url).json()
    poster_url = None

    if response.get("results"):
        poster_path = response["results"][0].get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"

    poster_cache[movie_name] = poster_url
    return poster_url

def get_recommendations(movie_name, similarity_matrix):
    """Get top 5 recommended movies."""
    movie_name = movie_name.lower()
    if movie_name not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = similarity_matrix[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        recommended_title = movies.iloc[i[0]]['title'].title()
        poster_url = fetch_movie_poster(recommended_title)
        recommendations.append({"title": recommended_title, "poster_url": poster_url})

    return recommendations

precomputed_recommendations = {}

def get_ensemble_recommendations(movie_name):
    """Get recommendations using ensemble learning."""
    movie_name = movie_name.lower()
    if movie_name in precomputed_recommendations:
        return precomputed_recommendations[movie_name]

    recommendations = []
    recommendations += get_recommendations(movie_name, similarity_tfidf)
    recommendations += get_recommendations(movie_name, similarity_lsi)
    recommendations += get_recommendations(movie_name, similarity_bm25)
    recommendations += get_recommendations(movie_name, similarity_word2vec)
    recommendations += get_recommendations(movie_name, similarity_jaccard)

    counter = Counter([rec["title"] for rec in recommendations])
    top_recommendations = counter.most_common(5)

    final_recommendations = []
    for rec in top_recommendations:
        title = rec[0]
        poster_url = fetch_movie_poster(title)
        final_recommendations.append({"title": title, "poster_url": poster_url})

    precomputed_recommendations[movie_name] = final_recommendations
    return final_recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation requests."""
    try:
        data = request.json
        movie_name = data.get('movie', None)

        if not movie_name:
            return jsonify({"error": "Missing 'movie' parameter"}), 400

        recommendations = get_ensemble_recommendations(movie_name)

        if not recommendations:
            return jsonify({"error": "Movie not found in dataset. Try another title."}), 404

        return jsonify({"movie": movie_name, "recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)