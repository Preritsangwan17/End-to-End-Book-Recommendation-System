import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from book_recommender_system.recommendation import Recommender


@pytest.fixture
def engine():
    pivot = pd.DataFrame([[5, 1], [5, 1], [1, 5]], index=["A", "B", "C"])
    ratings = pd.DataFrame({"title": ["A", "B", "C"], "image_url": ["a", "b", "c"]})
    return Recommender(
        NearestNeighbors(algorithm="brute").fit(csr_matrix(pivot)), pivot, ratings
    )


def test_excludes_selected_book_even_with_tied_distance(engine):
    results = engine.recommend("B", count=5)
    assert {r["title"] for r in results} == {"A", "C"}
    assert all(r["title"] != "B" for r in results)
    assert next(r for r in results if r["title"] == "A")["image_url"] == "a"


def test_unknown_title_has_clear_error(engine):
    with pytest.raises(ValueError, match="Unknown book"):
        engine.recommend("Missing")


def test_missing_artifacts_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python main.py"):
        Recommender.load(tmp_path)


def test_rejects_same_rows_but_different_feature_dimensions(engine):
    with pytest.raises(ValueError, match="do not match"):
        Recommender(engine.model, engine.pivot.assign(extra_user=0), engine.ratings)


def test_rejects_duplicate_titles(engine):
    duplicated = engine.pivot.copy()
    duplicated.index = ["A", "A", "C"]
    with pytest.raises(ValueError, match="unique"):
        Recommender(engine.model, duplicated, engine.ratings)
