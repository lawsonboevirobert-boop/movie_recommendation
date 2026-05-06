"""
vector_db.py
Couche d'abstraction pour la base vectorielle ChromaDB.
Utilisé par indexation.py et rag.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION  = "movies"
MODEL_NAME  = "all-MiniLM-L6-v2"
# ───────────────────────────────────────────────────────────────────────────────

# Initialisation unique (évite de recharger le modèle à chaque appel)
_model      = SentenceTransformer(MODEL_NAME)
_client     = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


# ── Écriture ────────────────────────────────────────────────────────────────────
def ajouter_documents(docs: list[dict]) -> None:
    """
    Encode et insère une liste de documents dans ChromaDB.

    Chaque document doit avoir la forme :
        {
            "id":       str,          # identifiant unique
            "texte":    str,          # texte à encoder
            "metadata": dict          # infos supplémentaires (optionnel)
        }
    """
    if not docs:
        print(" Aucun document à ajouter.")
        return

    ids       = [d["id"] for d in docs]
    textes    = [d["texte"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    embeddings = _model.encode(textes, show_progress_bar=False).tolist()

    _collection.add(
        ids        = ids,
        documents  = textes,
        embeddings = embeddings,
        metadatas  = metadatas,
    )
    print(f" {len(docs)} document(s) ajouté(s). Total : {_collection.count()}")


def ajouter_par_batch(docs: list[dict], batch_size: int = 128) -> None:
    """
    Version batch de ajouter_documents — recommandée pour les grands volumes.
    """
    from tqdm import tqdm

    for start in tqdm(range(0, len(docs), batch_size), desc="Indexation"):
        ajouter_documents(docs[start:start + batch_size])


# ── Lecture ─────────────────────────────────────────────────────────────────────
def rechercher(query: str, n_resultats: int = 5) -> list[dict]:
    """
    Recherche les documents les plus proches de la requête.

    Retourne une liste de dicts :
        {
            "texte":    str,
            "metadata": dict,
            "score":    float   # distance cosinus (plus petit = plus proche)
        }
    """
    embedding = _model.encode([query]).tolist()

    resultats = _collection.query(
        query_embeddings = embedding,
        n_results        = n_resultats,
    )

    return [
        {
            "texte":    resultats["documents"][0][i],
            "metadata": resultats["metadatas"][0][i],
            "score":    resultats["distances"][0][i],
        }
        for i in range(len(resultats["documents"][0]))
    ]


def get_par_id(doc_id: str) -> dict | None:
    """Récupère un document précis par son identifiant."""
    result = _collection.get(ids=[doc_id])
    if not result["documents"]:
        return None
    return {
        "texte":    result["documents"][0],
        "metadata": result["metadatas"][0],
    }


def compter() -> int:
    """Retourne le nombre de documents dans la collection."""
    return _collection.count()


# ── Suppression ──────────────────────────────────────────────────────────────────
def supprimer_document(doc_id: str) -> None:
    """Supprime un document par son identifiant."""
    _collection.delete(ids=[doc_id])
    print(f" Document '{doc_id}' supprimé.")


def supprimer_collection() -> None:
    """Supprime entièrement la collection (irréversible)."""
    _client.delete_collection(COLLECTION)
    print(f" Collection '{COLLECTION}' supprimée.")


# ── Test rapide ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f" Collection '{COLLECTION}' — {compter()} document(s)\n")

    # Exemple d'ajout
    test_docs = [
        {
            "id":       "test_1",
            "texte":    "Un film de science-fiction sur des voyages dans l'espace.",
            "metadata": {"title": "Test Film", "genre": "Sci-Fi"},
        }
    ]
    ajouter_documents(test_docs)

    # Exemple de recherche
    resultats = rechercher("aventure spatiale", n_resultats=3)
    print("\n Résultats pour 'aventure spatiale' :")
    for r in resultats:
        print(f"  [{r['score']:.4f}] {r['metadata'].get('title', '?')} — {r['texte'][:80]}…")

    # Nettoyage du doc de test
    supprimer_document("test_1")