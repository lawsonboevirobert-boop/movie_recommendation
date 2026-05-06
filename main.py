"""
main.py
Point d'entrée unique du projet.
Orchestre l'indexation et le système de questions-réponses.

Usage :
    python main.py                  # mode interactif (indexe si besoin)
    python main.py --reindex        # force une ré-indexation complète
    python main.py --query "..."    # pose une question directement
"""

import argparse
import os
import sys

from indexation import load_and_clean, index
from vector_db  import compter, rechercher
from rag        import ask


# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH    = "tmdb_5000_movies.csv"
CHROMA_PATH = "./chroma_db"
# ───────────────────────────────────────────────────────────────────────────────


def verifier_csv() -> bool:
    """Vérifie que le fichier CSV source est présent."""
    if not os.path.exists(CSV_PATH):
        print(f" Fichier introuvable : {CSV_PATH}")
        print("   Télécharge-le sur : https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        return False
    return True


def base_est_vide() -> bool:
    """Retourne True si la base vectorielle n'existe pas ou est vide."""
    try:
        return compter() == 0
    except Exception:
        return True


def etape_indexation(force: bool = False) -> None:
    """Lance l'indexation si nécessaire (ou forcée)."""
    if force:
        print(" Ré-indexation forcée...\n")
    elif base_est_vide():
        print("📭 Base vectorielle vide — indexation automatique...\n")
    else:
        n = compter()
        print(f" Base vectorielle prête ({n} films indexés). Indexation ignorée.")
        print("   (utilisez --reindex pour forcer une ré-indexation)\n")
        return

    if not verifier_csv():
        sys.exit(1)

    df = load_and_clean(CSV_PATH)
    index(df)
    print()


def mode_interactif() -> None:
    """Lance le chatbot de recommandation en boucle interactive."""
    print("=" * 60)
    print(" Système de recommandation de films — Mode interactif")
    print("=" * 60)
    print("Tapez votre question ou une commande :")
    print("  :quitter  — quitter le programme")
    print("  :reindex  — ré-indexer la base")
    print("  :stats    — afficher les stats de la base")
    print("-" * 60 + "\n")

    while True:
        try:
            query = input(" Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n À bientôt !")
            break

        if not query:
            continue

        # Commandes internes
        if query == ":quitter":
            print(" À bientôt !")
            break

        if query == ":reindex":
            if verifier_csv():
                df = load_and_clean(CSV_PATH)
                index(df)
            continue

        if query == ":stats":
            n = compter()
            print(f" Films indexés : {n}")
            print(f"   Base stockée  : {CHROMA_PATH}/\n")
            continue

        # Recherche + génération
        print("\n Recherche en cours...\n")

        # Affichage des films récupérés (retrieval)
        films = rechercher(query, n_resultats=5)
        print("📽️  Films récupérés :")
        for f in films:
            print(f"  [{f['score']:.3f}] {f['metadata']['title']} "
                  f"({f['metadata']['release_date'][:4]}) — "
                  f"{f['metadata']['genres']}")

        # Génération de la réponse
        print("\n Réponse :\n")
        reponse = ask(query, verbose=False)
        print(reponse)
        print("\n" + "-" * 60 + "\n")


def mode_query(query: str) -> None:
    """Pose une seule question et affiche la réponse, puis quitte."""
    print(f" Question : {query}\n")
    reponse = ask(query, verbose=True)
    print(f"\n Réponse :\n{reponse}")


# ── Point d'entrée ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Système RAG de recommandation de films TMDB"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force une ré-indexation complète de la base vectorielle"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Pose une question directement sans mode interactif"
    )
    args = parser.parse_args()

    # 1. Indexation (automatique ou forcée)
    etape_indexation(force=args.reindex)

    # 2. Mode d'utilisation
    if args.query:
        mode_query(args.query)
    else:
        mode_interactif()