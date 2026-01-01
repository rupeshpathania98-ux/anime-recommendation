from flask import Flask, request, jsonify
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ---------------------------
# ANIME DATABASE
# ---------------------------
ANIME_DATABASE = [

    # ---------------- ACTION / BATTLE ----------------
    {"title": "Attack on Titan", "genres": "Action, Drama, War, Dark", "rating": 9.0},
    {"title": "Demon Slayer", "genres": "Action, Supernatural, Historical", "rating": 8.7},
    {"title": "Jujutsu Kaisen", "genres": "Action, Supernatural, Dark", "rating": 8.6},
    {"title": "Naruto", "genres": "Action, Adventure, Ninja", "rating": 8.3},
    {"title": "One Punch Man", "genres": "Action, Comedy, Superhero", "rating": 8.7},

    # ---------------- DRAMA / EMOTIONAL ----------------
    {"title": "Grave of the Fireflies", "genres": "Drama, War, Tragedy, Emotional", "rating": 8.5},
    {"title": "A Silent Voice", "genres": "Drama, Romance, School, Emotional", "rating": 8.9},
    {"title": "Violet Evergarden", "genres": "Drama, Slice of Life, Emotional", "rating": 8.8},
    {"title": "Your Lie in April", "genres": "Drama, Romance, Music, Tragedy", "rating": 8.6},
    {"title": "Clannad After Story", "genres": "Drama, Romance, Family, Tragedy", "rating": 8.9},

    # ---------------- PSYCHOLOGICAL / DARK ----------------
    {"title": "Death Note", "genres": "Psychological, Thriller, Mystery", "rating": 9.0},
    {"title": "Tokyo Ghoul", "genres": "Action, Horror, Psychological, Dark", "rating": 7.8},
    {"title": "Parasyte", "genres": "Action, Horror, Psychological", "rating": 8.3},
    {"title": "Steins;Gate", "genres": "Sci-Fi, Thriller, Psychological", "rating": 9.1},
    {"title": "Monster", "genres": "Psychological, Crime, Thriller", "rating": 8.7},

    # ---------------- ROMANCE ----------------
    {"title": "Your Name", "genres": "Romance, Drama, Supernatural", "rating": 8.9},
    {"title": "Toradora", "genres": "Romance, Comedy, School", "rating": 8.1},
    {"title": "Horimiya", "genres": "Romance, Slice of Life, School", "rating": 8.2},
    {"title": "I Want to Eat Your Pancreas", "genres": "Romance, Drama, Tragedy", "rating": 8.6},
    {"title": "Weathering With You", "genres": "Romance, Fantasy, Drama", "rating": 8.3},

    # ---------------- FANTASY / SUPERNATURAL ----------------
    {"title": "Fullmetal Alchemist Brotherhood", "genres": "Action, Adventure, Fantasy, Drama", "rating": 9.2},
    {"title": "Hunter x Hunter", "genres": "Action, Adventure, Fantasy", "rating": 9.0},
    {"title": "Spirited Away", "genres": "Fantasy, Adventure, Family", "rating": 9.0},
    {"title": "Re:Zero", "genres": "Fantasy, Psychological, Isekai", "rating": 8.2},
    {"title": "Made in Abyss", "genres": "Fantasy, Adventure, Dark", "rating": 8.7},
]


# ---------------------------
# AI SETUP
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

anime_texts = [
    f"{anime['title']} {anime['genres']}"
    for anime in ANIME_DATABASE
]

anime_embeddings = model.encode(anime_texts)

# ---------------------------
# AI RECOMMENDER FUNCTION
# ---------------------------
def recommend_anime(prompt):
    prompt_embedding = model.encode([prompt])
    similarities = cosine_similarity(prompt_embedding, anime_embeddings)[0]

    top_indices = similarities.argsort()[-5:][::-1]

    return [ANIME_DATABASE[i] for i in top_indices]

# ---------------------------
# API ROUTE
# ---------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify([])

    results = recommend_anime(prompt)
    return jsonify(results)

# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
