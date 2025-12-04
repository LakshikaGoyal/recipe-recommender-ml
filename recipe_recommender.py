from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------------
# Sample Recipe Dataset (built-in)
# -------------------------------
data = {
    "title": [
        "Garlic Butter Chicken",
        "Creamy Chicken Pasta",
        "Egg Fried Rice",
        "Tomato Basil Pasta",
        "Veggie Salad",
        "Chocolate Mug Cake",
        "Garlic Cheese Toast"
    ],
    "ingredients": [
        "chicken, garlic, butter, salt, pepper",
        "chicken, pasta, cream, garlic, cheese",
        "rice, egg, spring onion, soy sauce, oil",
        "pasta, tomato, basil, garlic, olive oil",
        "lettuce, tomato, cucumber, olive oil, lemon",
        "flour, cocoa, sugar, milk, chocolate",
        "bread, garlic, cheese, butter"
    ]
}

df = pd.DataFrame(data)

# Combine title + ingredients for vectorization
df["combined"] = df["title"] + " " + df["ingredients"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# -------------------------------
# Recommend Function
# -------------------------------
def recommend_recipes(user_ingredients):
    user_tfidf = vectorizer.transform([user_ingredients])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Top 3 recipe indices sorted by similarity
    top_idx = similarities.argsort()[::-1][:3]

    results = []
    for idx in top_idx:
        results.append({
            "title": df.iloc[idx]["title"],
            "ingredients": df.iloc[idx]["ingredients"],
            "score": round(float(similarities[idx]), 3)
        })
    return results

# -------------------------------
# Flask UI (HTML inside Python)
# -------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Recipe Recommendation System</title>
    <style>
        body { font-family: Arial; padding: 40px; background: #fafafa; }
        h1 { color: #333; }
        .container { width: 50%; margin: auto; }
        input, button {
            padding: 12px;
            width: 100%;
            margin-top: 10px;
            border-radius: 6px;
            border: 1px solid #aaa;
        }
        .recipe {
            background: white;
            padding: 15px;
            margin-top: 10px;
            border-radius: 8px;
            box-shadow: 0 0 6px #ddd;
        }
        .score { font-size: 12px; color: gray; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍳 Recipe Recommendation System</h1>

        <form method="POST">
            <input name="ingredients" placeholder="Enter ingredients like: garlic, chicken, butter" required>
            <button type="submit">Recommend Recipes</button>
        </form>

        {% if results %}
            <h2>Recommendations:</h2>
            {% for r in results %}
                <div class="recipe">
                    <h3>{{ r.title }}</h3>
                    <p><b>Ingredients:</b> {{ r.ingredients }}</p>
                    <p class="score">Similarity: {{ r.score }}</p>
                </div>
            {% endfor %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        user_input = request.form["ingredients"]
        results = recommend_recipes(user_input)
    return render_template_string(HTML_PAGE, results=results)

if __name__ == "__main__":
    print("Starting Recipe Recommender System...")
    app.run(debug=True)
