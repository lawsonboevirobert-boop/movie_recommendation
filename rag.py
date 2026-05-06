"""
rag.py
Système RAG (Retrieval-Augmented Generation) pour recommandation de films.
- Retrieval : ChromaDB + sentence-transformers
- Generation : Groq (LLaMA-3) via API
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "movies"
MODEL_NAME   = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.3-70b-versatile"
TOP_K        = 5          # nombre de films récupérés par le retriever
# ───────────────────────────────────────────────────────────────────────────────

load_dotenv()


# ── Initialisation (une seule fois au démarrage) ────────────────────────────────
print(" Chargement du modèle d'embeddings…")
_embed_model = SentenceTransformer(MODEL_NAME)

_chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
_collection  = _chroma.get_collection(COLLECTION)

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(" Système RAG prêt.\n")


# ── Étape 1 : Retrieval ─────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Encode la requête et cherche les top_k films les plus proches
    dans la base vectorielle.
    """
    embedding = _embed_model.encode([query]).tolist()
    results   = _collection.query(
        query_embeddings=embedding,
        n_results=top_k,
    )

    films = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        films.append({
            "title":        meta["title"],
            "release_date": meta["release_date"],
            "vote_average": meta["vote_average"],
            "genres":       meta["genres"],
            "overview":     meta["overview"],
            "score":        results["distances"][0][i],   # distance cosinus
        })
    return films


# ── Étape 2 : Génération ────────────────────────────────────────────────────────
def build_context(films: list[dict]) -> str:
    """Formate les films récupérés en contexte lisible pour le LLM."""
    lines = []
    for i, f in enumerate(films, 1):
        lines.append(
            f"{i}. **{f['title']}** ({f['release_date'][:4]}) — "
            f"Note : {f['vote_average']}/10 — Genres : {f['genres']}\n"
            f"   Synopsis : {f['overview']}"
        )
    return "\n\n".join(lines)


def generate(query: str, films: list[dict]) -> str:
    """Envoie la question + le contexte à Groq et retourne la réponse."""
    context = build_context(films)

    system_prompt = (
        "Tu es un expert en cinéma et un assistant de recommandation de films. "
        "Tu réponds toujours en français, de manière claire et enthousiaste. "
        "Appuie-toi UNIQUEMENT sur les films fournis dans le contexte pour répondre. "
        "Si aucun film ne correspond à la demande, dis-le honnêtement."
    )

    user_prompt = (
        f"Question de l'utilisateur : {query}\n\n"
        f"Films disponibles dans la base de données :\n{context}\n\n"
        "Réponds à la question en te basant sur ces films. "
        "Explique pourquoi chaque film recommandé correspond à la demande."
    )

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Interface principale ────────────────────────────────────────────────────────
def ask(query: str, verbose: bool = False) -> str:
    """
    Point d'entrée principal du système RAG.
    verbose=True affiche les films récupérés avant la réponse.
    """
    films = retrieve(query)

    if verbose:
        print(f"  Films récupérés pour '{query}' :")
        for f in films:
            print(f"  - {f['title']} ({f['release_date'][:4]}) | score={f['score']:.4f}")
        print()

    return generate(query, films)


# ── Mode interactif ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(" Système de recommandation de films (tapez 'quitter' pour sortir)\n")
    print("Exemples de questions :")
    print("  - Je veux un film de science-fiction avec de l'action")
    print("  - Recommande-moi un film d'horreur psychologique")
    print("  - Quel film regarder en famille avec des enfants ?\n")

    while True:
        query = input(" Votre question : ").strip()
        if not query:
            continue
        if query.lower() in ("quitter", "quit", "exit", "q"):
            print(" À bientôt !")
            break

        print("\n Recherche en cours…\n")
        response = ask(query, verbose=True)
        print(f" Réponse :\n{response}\n")
        print("─" * 60 + "\n")