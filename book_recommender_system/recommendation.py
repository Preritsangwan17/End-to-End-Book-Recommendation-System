"""Load matching artifacts and retrieve item-based nearest neighbours."""

from pathlib import Path
import pickle
import numpy as np
from scipy.sparse import csr_matrix


class Recommender:
    def __init__(self, model, pivot, ratings):
        if len(pivot) != model.n_samples_fit_:
            raise ValueError("Model and book pivot do not match. Retrain the pipeline.")
        self.model, self.pivot, self.ratings = model, pivot, ratings

    @classmethod
    def load(cls, root=Path("artifacts")):
        root = Path(root)
        paths = [
            root / "trained_model/model.pkl",
            root / "serialized_data/book_pivot.pkl",
            root / "serialized_data/final_rating.pkl",
        ]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(
                "Recommendation artifacts are missing. Run python main.py first."
            )
        objects = []
        for path in paths:
            with path.open("rb") as handle:
                objects.append(pickle.load(handle))
        return cls(*objects)

    def recommend(self, title, count=5):
        if title not in self.pivot.index:
            raise ValueError(f"Unknown book: {title}")
        if count < 1:
            raise ValueError("count must be positive")
        index = self.pivot.index.get_loc(title)
        distances, indices = self.model.kneighbors(
            csr_matrix(self.pivot.iloc[[index]].to_numpy()),
            n_neighbors=min(count + 1, len(self.pivot)),
        )
        posters = self.ratings.drop_duplicates("title").set_index("title")["image_url"]
        results = []
        for distance, position in zip(distances[0], indices[0]):
            name = self.pivot.index[position]
            if name == title:
                continue
            url = posters.get(name, "")
            results.append(
                {
                    "title": name,
                    "distance": float(distance),
                    "image_url": url if isinstance(url, str) else "",
                }
            )
        return results[:count]
