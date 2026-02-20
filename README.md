# Hybrid Recommendation System

A hybrid recommendation system implemented by me, combining content-based and collaborative filtering approaches, deployed on AWS using Docker, ECR, and EC2.

## Concept and Implementation

This project implements a recommendation engine using multiple techniques:

- **Popularity-based:** Simple top-N recommendations based on ratings.  
  **Advantage:** Easy to implement  
  **Disadvantage:** Not personalized

- **Content-based:** Recommends items similar to what the user has interacted with (e.g., machine learning videos).  
  **Limitation:** Only shows items from the same field.

- **Collaborative-based:**
  - **User-based:** Recommends items based on similar user preferences.
  - **Item-based:** Recommends items based on similarity between items.

- **Hybrid system:** Combines content-based and collaborative filtering using a weighted approach (e.g., 40% content-based, 60% collaborative).

### Choosing User-based vs Item-based

- If number of users > number of items → Item-based (rows: items, columns: users) e.g., Netflix.
- If number of items > number of users → User-based (rows: users, columns: items) e.g., Instagram.
- Hybrid system allows variety and personalized recommendations.

## Project Structure

```
├── data
│   ├── collab_filtered_data.csv
│   ├── interaction_matrix.npz
│   ├── track_ids.npy
│   ├── cleaned_data.csv
│   ├── transformed_data.npz
│   └── transformed_hybrid_data.npz
├── docs
├── models
├── notebooks
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.py
└── src
    ├── app.py
    ├── collaborative_based_filtering.py
    ├── content_based_filtering.py
    ├── hybrid_recommendations.py
    ├── data_cleaning.py
    ├── transform_filtered_data.py
    └── __init__.py
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/abubakarsaddique22/hybrid_recommander_system.git
   cd hybrid_recommander_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Docker Deployment

- **Base Image:** `python:3.12`
- **Working Directory:** `/app/`
- **Dependencies:** Installed via `requirements.txt`
- **Preprocessed Data:** All CSV, `.npz`, and `.npy` files are prepared locally and copied into the container:
  - `collab_filtered_data.csv`
  - `interaction_matrix.npz`
  - `track_ids.npy`
  - `cleaned_data.csv`
  - `transformed_data.npz`
  - `transformed_hybrid_data.npz`

- **Python Scripts:** All scripts for content-based, collaborative-based, hybrid filtering, data cleaning, and transformations are copied into the container.

- **Port Exposed:** `8501`
- **Start Command:**
```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
```

**Why this approach?**
- Avoids building or transforming data inside the Docker container.
- Ensures faster startup and reproducible results.
- Makes the Docker image smaller and cleaner.

## How It Works

- Combines **content-based filtering** and **collaborative filtering**.
- Weighted hybrid allows tuning between personalization and variety.
- Returns top-N recommendations based on user preferences and item similarity.

## Advantages

- Personalized recommendations.
- Variety of content.
- Scalable for large datasets.

## License

This project is licensed under the MIT License.

