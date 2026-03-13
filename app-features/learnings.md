# Learnings & Bug Fixes

## 1. Wrong Qdrant Cluster URL

**Error:** `UnexpectedResponse: 404 (Not Found)` on `get_collections()`

**Cause:** `QDRANT_URL` in `.env` pointed to an old/deleted Qdrant Cloud cluster (`us-west-1`).

**Fix:** Updated `.env` with the correct active cluster URL:
```
QDRANT_URL=https://b53676e8-da2c-4633-b649-c55d973e886d.us-east-1-1.aws.cloud.qdrant.io
```
Also updated `QDRANT_API_KEY` to match the new cluster.

---

## 2. Collection Name Mismatch

**Error:** `/list` returned "No documents indexed yet" even after ingesting.

**Cause:** Code used `COLLECTION_NAME = "career_docs"` but the intended collection on the new cluster was `career-docs`.

**Fix:** Updated `main.py`:
```python
COLLECTION_NAME = "career-docs"
```
The `ensure_collection()` function auto-creates it on first upload.

---

## 3. `QdrantClient` has no attribute `search`

**Error:** `AttributeError: 'QdrantClient' object has no attribute 'search'`

**Cause:** `qdrant-client` v1.x removed the `search()` method. The new API uses `query_points()`.

**Fix:** Updated `query_node()` in `main.py`:
```python
# Before
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=TOP_K,
    with_payload=True,
)

# After
response = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=TOP_K,
    with_payload=True,
)
results = response.points
```
