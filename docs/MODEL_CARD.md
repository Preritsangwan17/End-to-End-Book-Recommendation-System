# Model card: Item-based book recommender

## Purpose and scope

Retrieve books with nearby reader-rating vectors from a Book-Crossing catalog. This is item-based
collaborative filtering, not personalized user recommendation or a language-model reading assistant.

## Training and inputs

Semicolon/Latin-1 source CSVs are filtered to readers with more than 200 records and titles with at
least 50 records in the filtered population. Repeated reader/title pairs are removed. A dense title-by-user
pivot fills missing entries with zero, then a sparse representation fits brute-force Euclidean nearest
neighbours. A query is an exact catalog title; the selected title is excluded from returned recommendations.

## Artifacts and checks

Model, pivot and title/image metadata must come from the same training run. Loading validates sample
and feature dimensions and rejects duplicate title keys. Equal dimensions alone cannot prove matching
column identities: regenerate the whole artifact set together whenever data changes.

## Evaluation status

The staged pipeline runs on the committed CSVs. Regression tests cover tied distances, missing artifacts,
unknown titles and incompatible dimensions. **No held-out ranking-quality evaluation has been completed.**
There are no Recall@K/NDCG@K, personalization or user-satisfaction claims.

## Limitations

Zero can mean implicit feedback or missing data. Euclidean distance is sensitive to vector magnitude;
cosine comparison and explicit/implicit-feedback handling are useful future experiments. Cold-start users
and unseen titles are unsupported. The source dataset's redistribution terms are not independently recorded;
the existing MIT code license does not cover third-party data.

## Reproduction

```bash
python main.py
python -m streamlit run app.py
```

Only load trusted pickle artifacts. Use the catalog dropdown to select a title that exists in the trained data.
