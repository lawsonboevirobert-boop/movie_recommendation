"""
app.py
Frontend Streamlit — style Cinéma / Netflix
Lancer avec : streamlit run app.py
"""

import streamlit as st
from rag import ask, retrieve

# ── Config page ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Recommandation de films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS Netflix style ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

  /* Fond cinéma */
  .stApp {
    background-color: #0a0a0a;
    background-image:
      radial-gradient(ellipse at 20% 50%, rgba(229,9,20,0.08) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 20%, rgba(229,9,20,0.05) 0%, transparent 50%);
    color: #e5e5e5;
  }

  /* Masquer les éléments Streamlit par défaut */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 4rem; max-width: 1200px; }

  /* Titre principal */
  .cine-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 0.08em;
    color: #e50914;
    text-shadow: 0 0 60px rgba(229,9,20,0.4);
    margin: 0;
    line-height: 1;
  }
  .cine-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    color: #999;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 2.5rem;
  }

  /* Barre de recherche */
  .stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 4px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stTextInput > div > div > input:focus {
    border-color: #e50914 !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.25) !important;
  }
  .stTextInput > div > div > input::placeholder { color: #666 !important; }

  /* Bouton recherche */
  .stButton > button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2rem !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100%;
  }
  .stButton > button:hover {
    background: #b81d24 !important;
    transform: translateY(-1px) !important;
  }

  /* Carte film */
  .film-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
  }
  .film-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #e50914;
    border-radius: 3px 0 0 3px;
  }
  .film-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(229,9,20,0.4);
  }
  .film-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: #fff;
    margin: 0 0 0.3rem 0;
  }
  .film-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 0.6rem;
    display: flex;
    gap: 1rem;
    align-items: center;
  }
  .film-genres {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.7rem;
  }
  .genre-tag {
    background: rgba(229,9,20,0.15);
    border: 1px solid rgba(229,9,20,0.3);
    color: #ff6b6b;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .film-overview {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #aaa;
    line-height: 1.6;
    margin: 0;
  }
  .score-badge {
    background: rgba(229,9,20,0.2);
    border: 1px solid rgba(229,9,20,0.5);
    color: #e50914;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
  }
  .star-rating {
    color: #f5c518;
    font-size: 0.85rem;
  }

  /* Réponse IA */
  .ai-response {
    background: rgba(229,9,20,0.06);
    border: 1px solid rgba(229,9,20,0.2);
    border-radius: 6px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #ddd;
    line-height: 1.8;
  }
  .ai-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    color: #e50914;
    margin-bottom: 0.8rem;
  }

  /* Section labels */
  .section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.1em;
    color: #fff;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.1);
    margin-left: 0.5rem;
  }

  /* Exemples de requêtes */
  .example-chip {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: #bbb;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    margin: 0.3rem;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }
  .example-chip:hover {
    background: rgba(229,9,20,0.15);
    color: #fff;
    border-color: rgba(229,9,20,0.3);
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #e50914 !important; }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="cine-title">CINE AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="cine-subtitle">✦ Recommandation intelligente par IA ✦</p>', unsafe_allow_html=True)

# ── Exemples cliquables ──────────────────────────────────────────────────────────
exemples = [
    "Film de science-fiction avec des robots",
    "Thriller psychologique des années 90",
    "Comédie romantique feel-good",
    "Film d'horreur avec des zombies",
    "Aventure épique avec des batailles",
    "Drame familial touchant",
]

# Initialiser l'état de session
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None

st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
cols = st.columns(len(exemples))
for i, exemple in enumerate(exemples):
    with cols[i]:
        if st.button(exemple, key=f"ex_{i}", use_container_width=True):
            st.session_state.query = exemple
st.markdown('</div>', unsafe_allow_html=True)

# ── Barre de recherche ───────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="",
        value=st.session_state.query,
        placeholder="🎬  Décris le film que tu cherches...",
        key="search_input",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<div style='padding-top:0.1rem'>", unsafe_allow_html=True)
    search_clicked = st.button("Rechercher", key="search_btn")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Lancement de la recherche ────────────────────────────────────────────────────
if search_clicked and query.strip():
    st.session_state.query = query
    with st.spinner("Analyse en cours..."):
        films  = retrieve(query, top_k=5)
        reponse = ask(query, verbose=False)
    st.session_state.results = {"films": films, "reponse": reponse, "query": query}

# ── Affichage des résultats ──────────────────────────────────────────────────────
if st.session_state.results:
    data = st.session_state.results

    # Réponse IA
    st.markdown(f"""
    <div class="ai-response">
      <div class="ai-label"> Recommandation IA</div>
      {data['reponse'].replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # Films récupérés
    st.markdown('<div class="section-label">Films sélectionnés</div>', unsafe_allow_html=True)

    for film in data["films"]:
        annee  = str(film.get("release_date", ""))[:4]
        note   = film.get("vote_average", 0)
        genres = film.get("genres", "")
        apercu = film.get("overview", "")
        score  = film.get("score", 1)
        pertinence = max(0, int((1 - score) * 100))

        # Tags genres
        genre_tags = "".join(
            f'<span class="genre-tag">{g.strip()}</span>'
            for g in genres.split(",") if g.strip()
        )

        etoiles = "★" * int(round(note / 2)) + "☆" * (5 - int(round(note / 2)))

        st.markdown(f"""
        <div class="film-card">
          <div class="film-title">{film.get('title', 'Inconnu')}</div>
          <div class="film-meta">
            <span>📅 {annee}</span>
            <span class="star-rating">{etoiles}</span>
            <span style="color:#888">{note}/10</span>
            <span class="score-badge">Pertinence {pertinence}%</span>
          </div>
          <div class="film-genres">{genre_tags}</div>
          <p class="film-overview">{apercu}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#444; font-family:'Inter',sans-serif; font-size:0.75rem; letter-spacing:0.1em;">
  CINE AI &nbsp;✦&nbsp; Propulsé par ChromaDB · sentence-transformers · Groq LLaMA
</div>
""", unsafe_allow_html=True)