"""
indexation.py
Création de la base vectorielle ChromaDB à partir du dataset TMDB 5000 movies.
Utilise sentence-transformers (all-MiniLM-L6-v2) pour générer les embeddings.
"""

import pandas as pd
import ast
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH      = "tmdb_5000_movies.csv"
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "movies"
MODEL_NAME    = "all-MiniLM-L6-v2"
BATCH_SIZE    = 128          # documents traités à la fois (évite les OOM)
# ───────────────────────────────────────────────────────────────────────────────


def parse_json_names(cell: str) -> str:
    """Extrait les noms d'une colonne JSON imbriquée (genres, keywords…)."""
    try:
        items = ast.literal_eval(cell)
        return ", ".join(i["name"] for i in items)
    except Exception:
        return ""


def build_text(row: pd.Series) -> str:
    """
    Construit le texte indexé pour chaque film.
    On concatène : titre · tagline · synopsis · genres · keywords
    afin que la recherche sémantique soit riche.
    """
    parts = []
    if pd.notna(row["title"]):
        parts.append(f"Title: {row['title']}")
    if pd.notna(row["tagline"]) and row["tagline"]:
        parts.append(f"Tagline: {row['tagline']}")
    if pd.notna(row["overview"]) and row["overview"]:
        parts.append(f"Overview: {row['overview']}")
    genres = parse_json_names(row["genres"])
    if genres:
        parts.append(f"Genres: {genres}")
    keywords = parse_json_names(row["keywords"])
    if keywords:
        parts.append(f"Keywords: {keywords}")
    return " | ".join(parts)


def load_and_clean(path: str) -> pd.DataFrame:
    """Charge le CSV et filtre les lignes sans synopsis (inutilisables)."""
    df = pd.read_csv(path)
    df = df[df["overview"].notna() & (df["overview"].str.strip() != "")]
    df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)
    print(f" {len(df)} films chargés après nettoyage.")
    return df


def index(df: pd.DataFrame) -> None:
    """Encode les films et les insère dans ChromaDB par batchs."""
    print(f" Chargement du modèle '{MODEL_NAME}'…")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Repart de zéro si la collection existe déjà
    try:
        client.delete_collection(COLLECTION)
        print(" Ancienne collection supprimée.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # distance cosinus
    )

    texts     = [build_text(row) for _, row in df.iterrows()]
    ids       = [str(row["id"]) for _, row in df.iterrows()]
    metadatas = [
        {
            "title":        str(row["title"]),
            "release_date": str(row["release_date"]),
            "vote_average": float(row["vote_average"]),
            "genres":       parse_json_names(row["genres"]),
            "overview":     str(row["overview"])[:500],  # tronqué pour Chroma
        }
        for _, row in df.iterrows()
    ]

    print(f" Encodage + indexation de {len(texts)} films…")
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        end        = start + BATCH_SIZE
        batch_txt  = texts[start:end]
        embeddings = model.encode(batch_txt, show_progress_bar=False).tolist()

        collection.add(
            ids        = ids[start:end],
            documents  = batch_txt,
            embeddings = embeddings,
            metadatas  = metadatas[start:end],
        )

    print(f"\n Base vectorielle prête : {collection.count()} films indexés.")
    print(f"   Stockage : {CHROMA_PATH}/")


if __name__ == "__main__":
    df = load_and_clean(CSV_PATH)
    index(df)