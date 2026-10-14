"""Streamlit interface. Train artifacts with `python main.py` before launch."""

from pathlib import Path
import streamlit as st
from book_recommender_system.recommendation import Recommender

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
st.title("Book Recommender")
st.write("Discover books with similar reader-rating patterns.")


@st.cache_resource
def load_recommender(version):
    return Recommender.load(ROOT / "artifacts")


try:
    model_path = ROOT / "artifacts/trained_model/model.pkl"
    engine = load_recommender(model_path.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as error:
    st.info(str(error))
    st.code("python main.py")
    st.stop()

title = st.selectbox("Choose a book", engine.pivot.index.tolist())
if st.button("Find similar books", type="primary"):
    results = engine.recommend(title)
    if not results:
        st.info("The catalog needs more books to produce recommendations.")
    else:
        for column, result in zip(st.columns(len(results)), results):
            with column:
                st.subheader(result["title"])
                if result["image_url"].startswith(("https://", "http://")):
                    st.image(result["image_url"])
                st.caption(f"Neighbour distance: {result['distance']:.3f}")
st.caption(
    "Item-based collaborative filtering • Research demo • No personalization or ranking-quality claims"
)
