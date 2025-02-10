from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os

app = Flask(__name__)

# URLs for Pickle Files
MODEL_URLS = {
    "popular.pkl": "https://book-recommender-files.s3.us-east-1.amazonaws.com/popular.pkl",
    "pt.pkl": "https://book-recommender-files.s3.us-east-1.amazonaws.com/pt.pkl",
    "books.pkl": "https://book-recommender-files.s3.us-east-1.amazonaws.com/books.pkl",
    "similarity_scores.pkl": "https://book-recommender-files.s3.us-east-1.amazonaws.com/similarity_scores.pkl",
}

LOCAL_PATH = "models/"

# Ensure local directory exists
os.makedirs(LOCAL_PATH, exist_ok=True)

# Function to download pickle files from URLs
def download_pkl_files():
    for filename, url in MODEL_URLS.items():
        local_file = os.path.join(LOCAL_PATH, filename)
        if not os.path.exists(local_file):  # Download only if not already present
            print(f"Downloading {filename} from {url}...")
            response = requests.get(url)
            with open(local_file, 'wb') as f:
                f.write(response.content)

# Download files before app starts
download_pkl_files()

# Load the Pickle Files
popular_df = pickle.load(open(os.path.join(LOCAL_PATH, "popular.pkl"), "rb"))
pt = pickle.load(open(os.path.join(LOCAL_PATH, "pt.pkl"), "rb"))
books = pickle.load(open(os.path.join(LOCAL_PATH, "books.pkl"), "rb"))
similarity_scores = pickle.load(open(os.path.join(LOCAL_PATH, "similarity_scores.pkl"), "rb"))

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item = [
            temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
            temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
            temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
        ]
        data.append(item)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
