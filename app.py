"""
Football Value Bet Dashboard — Streamlit app
"""

import os
import datetime
import json
import streamlit as st
from analyzer import (
    FootballDataClient, OddsClient,
    expected_goals, match_probabilities, detect_value_bets, teams_match,
    compute_team_stats_from_matches, compute_form_from_fd_matches, demo_injury,
    remove_overround, kelly_fraction,
    LEAGUES, SPORT_KEYS, MIN_GAMES_HOME, MIN_GAMES_AWAY, MIN_GAMES_INTL,
    FD_COMPETITIONS, FD_WC_ID,
    DEMO_MATCHES, DEMO_RECENT, DEMO_ODDS, _NO_INJ,
    current_season,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Value Bet Analyzer",
    page_icon="⚽",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif; }
  .value-pill {
    background: #16a34a22; border: 1px solid #16a34a; color: #4ade80;
    border-radius: 999px; padding: 2px 12px; font-size: 0.78rem;
    font-weight: 600; display: inline-block;
  }
  .no-value-pill {
    background: #ffffff0a; border: 1px solid #334155; color: #64748b;
    border-radius: 999px; padding: 2px 12px; font-size: 0.78rem;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Football Value Bet Analyzer")
st.markdown(f"**{datetime.datetime.now().strftime('%A, %d %B %Y')}** — Today's matches with market edge")
st.divider()

# ── API keys — st.secrets (Streamlit Cloud) with os.getenv fallback (local) ──
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

AF_KEY   = _get_secret("FOOTBALL_DATA_API_KEY", "YOUR_API_FOOTBALL_KEY")
ODDS_KEY = _get_secret("ODDS_API_KEY",          "YOUR_ODDS_API_KEY")

if "force_demo" not in st.session_state:
    st.session_state.force_demo = False

DEMO = (AF_KEY == "YOUR_API_FOOTBALL_KEY" or ODDS_KEY == "YOUR_ODDS_API_KEY"
        or st.session_state.force_demo)

if DEMO:
    col_dm, col_dm2 = st.columns([5, 1])
    col_dm.info("ℹ️ **Demo mode** — set secrets `FOOTBALL_DATA_API_KEY` and `ODDS_API_KEY` for live data.", icon="ℹ️")
    if st.session_state.force_demo:
        if col_dm2.button("❌ Salir demo"):
            st.session_state.force_demo = False
            st.cache_data.clear()
            st.rerun()


# ── Analysis ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def run_analysis(demo: bool, fd_key: str, odds_key: str):
    debug = {"mode": "demo" if demo else "live", "api_status": None,
             "raw_match_count": None, "tracked_match_count": None, "error": None}

    if demo:
        today_matches = DEMO_MATCHES
        all_odds      = DEMO_ODDS
        fd_client     = None
        odds_client   = None
        debug["raw_match_count"] = len(today_matches)
        debug["tracked_match_count"] = len(today_matches)
    else:
        import requests as _req
        import datetime as _dt
        _today    = _dt.date.today()
        today_s   = _today.isoformat()
        tomorrow_s = (_today + _dt.timedelta(days=1)).isoformat()
        fd_client  = FootballDataClient(fd_key)
        fd_client._matches_cache = {}
        try:
            # football-data.org returns empty when dateFrom==dateTo → use tomorrow and filter
            resp = _req.get(
                f"{__import__('analyzer').FD_BASE}/matches",
                headers={"X-Auth-Token": fd_key},
                params={"dateFrom": today_s, "dateTo": tomorrow_s},
                timeout=15,
            )
            debug["api_status"] = resp.status_code
            if resp.status_code != 200:
                debug["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
                today_matches = []
            else:
                all_m = resp.json().get("matches", [])
                debug["raw_match_count"] = len(all_m)
                today_matches = [
                    m for m in all_m
                    if m.get("utcDate", "").startswith(today_s)
                    and m.get("competition", {}).get("id") in FD_COMPETITIONS
                ]
                debug["tracked_match_count"] = len(today_matches)
        except Exception as e:
            debug["error"] = str(e)
            today_matches = []

        odds_client = OddsClient(odds_key)
        all_odds = []
        for sk in SPORT_KEYS:
            all_odds.extend(odds_client.get_odds(sk))

    results_by_league: dict[str, list] = {}

    for match in today_matches:
        comp_id   = match["competition"]["id"]
        comp_name = FD_COMPETITIONS.get(comp_id, "Unknown")
        is_wc     = (comp_id == FD_WC_ID)

        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]
        home_id   = match["homeTeam"]["id"]
        away_id   = match["awayTeam"]["id"]
        kickoff   = match["utcDate"][:16].replace("T", " ") + " UTC"

        # ── Recent matches → stats + form
        if DEMO:
            home_recent = DEMO_RECENT.get(home_id, [])
            away_recent = DEMO_RECENT.get(away_id, [])
        else:
            home_recent = fd_client.get_team_recent_matches(home_id)
            away_recent = fd_client.get_team_recent_matches(away_id)

        home_stats = compute_team_stats_from_matches(home_recent, home_id)
        away_stats = compute_team_stats_from_matches(away_recent, away_id)
        home_form  = compute_form_from_fd_matches(home_recent, home_id)
        away_form  = compute_form_from_fd_matches(away_recent, away_id)
        home_inj   = _NO_INJ
        away_inj   = _NO_INJ

        played_key   = "total" if is_wc else "home"
        played_key_a = "total" if is_wc else "away"
        min_h        = MIN_GAMES_INTL if is_wc else MIN_GAMES_HOME
        min_a        = MIN_GAMES_INTL if is_wc else MIN_GAMES_AWAY

        home_games = home_stats.get("fixtures", {}).get("played", {}).get(played_key, 0)
        away_games = away_stats.get("fixtures", {}).get("played", {}).get(played_key_a, 0)
        insufficient_data = home_games < min_h or away_games < min_a

        if insufficient_data:
            lam_h, lam_a, probs  = None, None, None
            best, bets, has_odds = {}, [], False
        else:
            lam_h, lam_a = expected_goals(
                home_stats, away_stats,
                home_form=home_form, away_form=away_form,
                home_injury=home_inj, away_injury=away_inj,
                is_neutral=is_wc,
            )
            probs = match_probabilities(lam_h, lam_a)

            matched_event = next(
                (ev for ev in all_odds
                 if teams_match(home_name, away_name,
                                ev.get("home_team", ""), ev.get("away_team", ""))),
                None,
            )
            if matched_event:
                best     = (_demo_best_odds(matched_event, home_name, away_name)
                            if demo else odds_client.best_odds(matched_event))
                bets     = detect_value_bets(probs, best)
                has_odds = True
            else:
                best, bets, has_odds = {}, [], False

        results_by_league.setdefault(comp_name, []).append({
            "home_team":         home_name,
            "away_team":         away_name,
            "competition":       comp_name,
            "kickoff":           kickoff,
            "lambda_home":       lam_h,
            "lambda_away":       lam_a,
            "model_probs":       probs,
            "best_odds":         best,
            "value_bets":        bets,
            "has_odds":          has_odds,
            "insufficient_data": insufficient_data,
            "home_form":         home_form,
            "away_form":         away_form,
            "home_injuries":     home_inj,
            "away_injuries":     away_inj,
            "is_neutral":        is_wc,
        })

    return results_by_league, debug


def _demo_best_odds(event: dict, home_name: str, away_name: str) -> dict:
    best = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for bm in event.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            for o in mkt.get("outcomes", []):
                n, p = o.get("name", ""), o.get("price", 0)
                if n == home_name:   best["home"] = max(best["home"], p)
                elif n == away_name: best["away"] = max(best["away"], p)
                elif n == "Draw":    best["draw"] = max(best["draw"], p)
    return best


# ── UI helpers ────────────────────────────────────────────────────────────────
def _form_badge(result: str) -> str:
    c = {"W": "#16a34a", "D": "#d97706", "L": "#dc2626"}.get(result, "#475569")
    return (f'<span style="background:{c};color:white;border-radius:4px;'
            f'padding:1px 7px;font-size:0.72rem;font-weight:700;margin-right:3px;">{result}</span>')


def _render_form_injuries(r: dict):
    hf = r.get("home_form", {})
    af = r.get("away_form", {})
    hi = r.get("home_injuries", {})
    ai = r.get("away_injuries", {})

    if hf.get("results") or af.get("results"):
        c1, c2 = st.columns(2)
        with c1:
            if hf.get("results"):
                badges = "".join(_form_badge(x) for x in hf["results"])
                st.markdown(f"Forma {r['home_team']}: {badges}", unsafe_allow_html=True)
        with c2:
            if af.get("results"):
                badges = "".join(_form_badge(x) for x in af["results"])
                st.markdown(f"Forma {r['away_team']}: {badges}", unsafe_allow_html=True)

    hi_p = hi.get("players", [])
    ai_p = ai.get("players", [])
    if hi_p or ai_p:
        st.markdown("**⚕️ Bajas / Suspensiones**")
        c1, c2 = st.columns(2)
        with c1:
            for p in hi_p:
                st.caption(f"🔴 {p['name']} ({p['position']}) — {p['reason']}")
        with c2:
            for p in ai_p:
                st.caption(f"🔴 {p['name']} ({p['position']}) — {p['reason']}")

    if r.get("is_neutral"):
        st.caption("🌍 Campo neutral — sin ventaja local · modelo usa estadísticas globales")


# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner("Analysing matches…"):
    results_by_league, _dbg = run_analysis(DEMO, AF_KEY, ODDS_KEY)

if not results_by_league:
    if _dbg.get("error"):
        st.error(f"❌ API error: {_dbg['error']}")
    elif _dbg.get("api_status") and _dbg["api_status"] != 200:
        st.error(f"❌ football-data.org returned HTTP {_dbg['api_status']}")
    else:
        raw = _dbg.get("raw_match_count")
        tracked = _dbg.get("tracked_match_count")
        if raw is not None:
            st.warning(
                f"No hay partidos hoy en las ligas monitorizadas.\n\n"
                f"La API devolvió **{raw} partidos** en total, "
                f"**{tracked}** en las competiciones seguidas (PL, LaLiga, Serie A, Bundesliga, Ligue 1, UCL, UEL, WC).\n\n"
                f"Es posible que hoy no haya jornada. Prueba mañana o activa el modo demo con el botón de abajo."
            )
        else:
            st.warning("No matches scheduled today or insufficient historical data.")
    if st.button("📺 Ver datos de ejemplo (Demo)"):
        st.session_state.force_demo = True
        st.cache_data.clear()
        st.rerun()
    st.stop()

all_matches = [m for matches in results_by_league.values() for m in matches]

# ── KPI row ───────────────────────────────────────────────────────────────────
modelled   = [r for r in all_matches if not r["insufficient_data"]]
total_bets = sum(len(r["value_bets"]) for r in modelled)
best_edge  = max((b["edge"] for r in modelled for b in r["value_bets"]), default=0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matches today",   len(all_matches))
k2.metric("Fully analysed",  len(modelled))
k3.metric("Value bets found", total_bets)
k4.metric("Best edge",        f"+{best_edge:.1%}" if best_edge else "—")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About the model")
    st.markdown(
        "**Dixon-Coles model** (Poisson bivariante corregido, ρ = −0.13):\n\n"
        "- **Stats históricas** (hasta 38 partidos / 2 temporadas)\n"
        "- **Forma reciente** (últimos 5 partidos, peso 35%)\n"
        "- **Corrección baja puntuación** via τ(i,j) — DC vs Poisson simple\n\n"
        "Edge calculado vs probabilidad de mercado **sin overround** (normalización Pinnacle).\n\n"
        "Umbrales: FUERTE ≥ 7% edge + EV ≥ 10% | MODERADO 4–7% | DÉBIL < 4% → no apostar.\n\n"
        "Kelly fraccionado: 25% (moderado) · 50% (fuerte, datos completos). Máx 3% bankroll."
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 🔍 Estado")
    st.markdown(f"**Modo:** {'Demo' if DEMO else 'Live (API real)'}")
    st.markdown(f"**Clave FD:** {'✅ configurada' if AF_KEY != 'YOUR_API_FOOTBALL_KEY' else '❌ no configurada'}")
    st.markdown(f"**Clave Odds:** {'✅ configurada' if ODDS_KEY != 'YOUR_ODDS_API_KEY' else '❌ no configurada'}")
    if not DEMO and results_by_league:
        raw = _dbg.get("raw_match_count", "?")
        tracked = _dbg.get("tracked_match_count", "?")
        st.markdown(f"**Partidos API:** {raw} totales, {tracked} en ligas seguidas")


# ── Value paragraph ───────────────────────────────────────────────────────────
def _value_paragraph(r: dict) -> str:
    best_bet = max(r["value_bets"], key=lambda b: b["edge"])
    outcome  = best_bet["outcome"]
    home, away = r["home_team"], r["away_team"]
    lh = r["lambda_home"] or 0
    la = r["lambda_away"] or 0
    edge = best_bet["edge"]
    diff = lh - la

    if diff > 0.4:
        xg_ctx = f"{home} llega con un perfil ofensivo claramente superior según el modelo, lo que hace que la cuota actual subestime significativamente sus opciones reales."
    elif diff < -0.4:
        xg_ctx = f"{away} presenta un perfil ofensivo superior, algo que el mercado habitualmente penaliza más de lo que los datos justifican."
    elif abs(diff) <= 0.15:
        xg_ctx = f"Ambos equipos llegan con perfiles ofensivos muy similares, lo que hace al mercado especialmente propenso a mispricing cuando polariza hacia un resultado concreto."
    elif diff > 0:
        xg_ctx = f"El modelo otorga una ventaja ofensiva moderada a {home} que el mercado no refleja del todo en su estructura de precios."
    else:
        xg_ctx = f"El modelo detecta una ligera superioridad ofensiva de {away} que los bookmakers están infravalorando."

    if "Home" in outcome:
        bias = "Los bookmakers suelen sobrevalorar las opciones de empate y sorpresa visitante, dejando la victoria local a un precio superior a su valor real."
    elif "Away" in outcome:
        bias = "La ventaja de campo está sistemáticamente sobrevalorada en los mercados: cuando el visitante tiene las métricas a favor, la cuota tiende a estar inflada."
    else:
        bias = "Los mercados tienden a infraponderar el empate en partidos equilibrados, concentrando liquidez en las victorias."

    size = "pronunciada" if edge > 0.12 else "clara" if edge > 0.07 else "moderada"
    return (f"{xg_ctx} {bias} La ineficiencia detectada es {size} y consistente con el "
            f"tipo de partido, lo que la convierte en una oportunidad con fundamento estadístico sólido.")


# ── Bet history helpers ───────────────────────────────────────────────────────
_BET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bets_history.json")

def _load_history() -> list:
    if not os.path.exists(_BET_FILE):
        return []
    try:
        with open(_BET_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(data: list):
    with open(_BET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _register_bet(bet: dict) -> bool:
    history = _load_history()
    today   = datetime.datetime.now().strftime("%Y-%m-%d")
    if any(h["date"] == today and h["match"] == bet.get("match") and h["outcome"] == bet.get("outcome")
           for h in history):
        return False
    history.append({**bet, "date": today, "result": None})
    _save_history(history)
    return True

def _mark_result(idx: int, result: str):
    history = _load_history()
    if 0 <= idx < len(history):
        history[idx]["result"] = result
        _save_history(history)


# ── Mejor apuesta del día ─────────────────────────────────────────────────────
_enriched_bets = [
    {**b,
     "match":       f"{r['home_team']} vs {r['away_team']}",
     "competition": r.get("competition", ""),
     "kickoff":     r.get("kickoff", "")}
    for r in all_matches
    for b in r.get("value_bets", [])
]
_best = max(_enriched_bets, key=lambda b: b["edge"]) if _enriched_bets else None

st.markdown("### 🏆 Mejor Apuesta del Día")
if _best:
    _icon = {"FUERTE": "🟢", "MODERADO": "🟡", "DÉBIL": "🔴"}.get(_best.get("edge_label", ""), "⚪")
    _col_info, _col_stats = st.columns([3, 2])
    with _col_info:
        st.markdown(f"#### {_best['match']}")
        st.caption(f"🕐 {_best['kickoff']}  ·  {_best['competition']}")
        st.markdown(f"**Pick:** {_best['outcome']}  ·  **Cuota:** {_best['best_odds']:.2f}")
    with _col_stats:
        _s1, _s2, _s3 = st.columns(3)
        _s1.metric("Prob. modelo", f"{_best['model_prob']:.1%}")
        _s2.metric(f"Edge {_icon}", f"+{_best['edge']:.1%}")
        _s3.metric("EV", f"{_best.get('ev', 0):+.1%}")
        if _best.get("kelly"):
            st.caption(f"Kelly: **{_best['kelly']:.2%}** del bankroll")
    if st.button("➕ Registrar esta apuesta", key="reg_best"):
        if _register_bet(_best):
            st.success("✅ Apuesta registrada correctamente")
        else:
            st.info("ℹ️ Ya estaba registrada hoy")
else:
    st.info("No hay apuestas con value detectadas hoy.")

st.divider()


# ── Registro de apuestas ──────────────────────────────────────────────────────
st.markdown("### 📊 Registro de Apuestas")

_history = _load_history()
_won     = [h for h in _history if h.get("result") == "W"]
_lost    = [h for h in _history if h.get("result") == "L"]
_pending = [h for h in _history if h.get("result") is None]
_settled = len(_won) + len(_lost)
_win_pct = len(_won) / _settled if _settled else 0

_k1, _k2, _k3, _k4 = st.columns(4)
_k1.metric("Total registradas", len(_history))
_k2.metric("Ganadas ✅",        len(_won))
_k3.metric("Perdidas ❌",       len(_lost))
_k4.metric("% Acierto",         f"{_win_pct:.1%}" if _settled else "—")

if _settled > 0:
    st.progress(_win_pct, text=f"{len(_won)}/{_settled} apuestas resueltas")

if _pending:
    st.markdown("**⏳ Pendientes — marca el resultado cuando lo conozcas:**")
    for _real_i, _h in [(i, h) for i, h in enumerate(_history) if h.get("result") is None]:
        _c1, _c2, _c3, _c4, _c5 = st.columns([3, 1, 1, 1, 1])
        _c1.markdown(f"**{_h['match']}** — {_h['outcome']} @ {float(_h.get('best_odds', 0)):.2f}")
        _c2.caption(_h.get("date", ""))
        _c3.caption(f"+{_h.get('edge', 0):.1%}")
        if _c4.button("✅ G", key=f"win_{_real_i}"):
            _mark_result(_real_i, "W")
            st.rerun()
        if _c5.button("❌ P", key=f"loss_{_real_i}"):
            _mark_result(_real_i, "L")
            st.rerun()

_settled_list = [h for h in _history if h.get("result") in ("W", "L")]
if _settled_list:
    with st.expander(f"📋 Historial completo ({len(_settled_list)} apuestas resueltas)"):
        for _h in reversed(_settled_list):
            _ri = "✅" if _h["result"] == "W" else "❌"
            st.caption(
                f"{_ri} {_h.get('date', '')} · {_h['match']} — "
                f"{_h['outcome']} @ {float(_h.get('best_odds', 0)):.2f} · "
                f"Edge +{_h.get('edge', 0):.1%}"
            )

st.divider()


# ── Top 3 value bets ──────────────────────────────────────────────────────────
top3_matches = sorted(
    [r for r in all_matches if r["value_bets"]],
    key=lambda r: max(b["edge"] for b in r["value_bets"]),
    reverse=True,
)[:3]

if top3_matches:
    st.markdown("### 🏆 Top 3 Value Bets")
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(top3_matches):
        probs = r["model_probs"]
        col_title, col_badge = st.columns([5, 1])
        with col_title:
            st.markdown(f"#### {medals[i]} {r['home_team']} vs {r['away_team']}")
            st.caption(f"🕐 {r['kickoff']}  ·  {r['competition']}")
        with col_badge:
            st.markdown('<span class="value-pill">🎯 VALUE</span>', unsafe_allow_html=True)

        _render_form_injuries(r)

        col_xg, col_probs = st.columns([2, 3])
        with col_xg:
            st.markdown("**Expected Goals**")
            st.markdown(f"🏠 {r['home_team']}: **{r['lambda_home']}**")
            st.markdown(f"✈️ {r['away_team']}: **{r['lambda_away']}**")
        with col_probs:
            st.markdown("**Probabilidades Dixon-Coles**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Local", f"{probs['home_win']:.1%}")
            c2.metric("Empate", f"{probs['draw']:.1%}")
            c3.metric("Visitante", f"{probs['away_win']:.1%}")
            c4, c5 = st.columns(2)
            c4.metric("Más de 2.5", f"{probs.get('over_2_5', 0):.1%}")
            c5.metric("Ambos marcan", f"{probs.get('btts', 0):.1%}")
            for _n, _dc, _ps in [("Local", "home_win", "poisson_home_win"),
                                  ("Empate", "draw", "poisson_draw"),
                                  ("Visit.", "away_win", "poisson_away_win")]:
                _d = abs(probs.get(_dc, 0) - probs.get(_ps, 0))
                if _d > 0.02:
                    st.caption(f"⚠️ DC vs Poisson — {_n}: {probs[_dc]:.1%} vs {probs[_ps]:.1%} (Δ{_d:.1%})")

        if r["has_odds"] and r["best_odds"]:
            st.markdown("**Best Available Odds**")
            o1, o2, o3 = st.columns(3)
            o1.metric("Home", f"{r['best_odds'].get('home', 0):.2f}")
            o2.metric("Draw", f"{r['best_odds'].get('draw', 0):.2f}")
            o3.metric("Away", f"{r['best_odds'].get('away', 0):.2f}")

        st.markdown("**Value Bets Detected**")
        for b in r["value_bets"]:
            _icon = {"FUERTE": "🟢", "MODERADO": "🟡", "DÉBIL": "🔴"}.get(b.get("edge_label", ""), "⚪")
            b1, b2, b3, b4, b5 = st.columns([2, 1, 1, 1, 2])
            b1.markdown(f"🎯 **{b['outcome']}** {_icon} `{b.get('edge_label','')}`")
            b2.markdown(f"Odds: **{b['best_odds']:.2f}**")
            b3.markdown(f"Modelo DC: **{b['model_prob']:.1%}**")
            b4.markdown(f"Mkt limpio: **{b.get('clean_prob', b['implied_prob']):.1%}**")
            b5.markdown(f"Edge: **+{b['edge']:.1%}** | EV: **{b.get('ev', 0):+.1%}**")
            if b.get("kelly"):
                st.caption(f"Kelly fraccionado: **{b['kelly']:.2%}** del bankroll")
            st.progress(min(int(b["edge"] * 100), 100))

        st.markdown(f"> *{_value_paragraph(r)}*")
        st.divider()

# ── All matches by league ─────────────────────────────────────────────────────
LEAGUE_FLAG = {
    "Premier League":   "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":          "🇪🇸",
    "Serie A":          "🇮🇹",
    "Bundesliga":       "🇩🇪",
    "Ligue 1":          "🇫🇷",
    "Champions League": "🏆",
    "Europa League":    "🌍",
    "World Cup":        "🌍🏆",
}

st.markdown("### 📊 Análisis de Todos los Partidos")

sorted_leagues = sorted(
    results_by_league.items(),
    key=lambda kv: -sum(len(m["value_bets"]) for m in kv[1])
)

for league_name, matches in sorted_leagues:
    flag  = LEAGUE_FLAG.get(league_name, "⚽")
    n_vb  = sum(len(m["value_bets"]) for m in matches)
    badge = f"🎯 {n_vb} value bet{'s' if n_vb != 1 else ''}" if n_vb else "No value bets"

    with st.expander(
        f"{flag} **{league_name}** — {len(matches)} match{'es' if len(matches) != 1 else ''}  ·  {badge}",
        expanded=n_vb > 0,
    ):
        for r in matches:
            has_bets = bool(r["value_bets"])
            probs    = r["model_probs"]

            col_title, col_badge = st.columns([5, 1])
            with col_title:
                st.markdown(f"#### {r['home_team']} vs {r['away_team']}")
                st.caption(f"🕐 {r['kickoff']}")
            with col_badge:
                if r["insufficient_data"]:
                    st.markdown('<span class="no-value-pill">Sin datos</span>', unsafe_allow_html=True)
                elif has_bets:
                    st.markdown('<span class="value-pill">🎯 VALUE</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="no-value-pill">No edge</span>', unsafe_allow_html=True)

            if r["insufficient_data"]:
                st.caption("⚠️ No hay suficiente información para el análisis de este partido.")
            else:
                _render_form_injuries(r)

                col_xg, col_probs = st.columns([2, 3])
                with col_xg:
                    st.markdown("**Expected Goals**")
                    st.markdown(f"🏠 {r['home_team']}: **{r['lambda_home']}**")
                    st.markdown(f"✈️ {r['away_team']}: **{r['lambda_away']}**")
                with col_probs:
                    st.markdown("**Probabilidades Dixon-Coles**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Local", f"{probs['home_win']:.1%}")
                    c2.metric("Empate", f"{probs['draw']:.1%}")
                    c3.metric("Visitante", f"{probs['away_win']:.1%}")
                    c4, c5 = st.columns(2)
                    c4.metric("Más de 2.5", f"{probs.get('over_2_5', 0):.1%}")
                    c5.metric("Ambos marcan", f"{probs.get('btts', 0):.1%}")
                    for _n, _dc, _ps in [("Local", "home_win", "poisson_home_win"),
                                          ("Empate", "draw", "poisson_draw"),
                                          ("Visit.", "away_win", "poisson_away_win")]:
                        _d = abs(probs.get(_dc, 0) - probs.get(_ps, 0))
                        if _d > 0.02:
                            st.caption(f"⚠️ DC vs Poisson — {_n}: {probs[_dc]:.1%} vs {probs[_ps]:.1%} (Δ{_d:.1%})")

                if r["has_odds"] and r["best_odds"]:
                    st.markdown("**Best Available Odds**")
                    o1, o2, o3 = st.columns(3)
                    o1.metric("Home", f"{r['best_odds'].get('home', 0):.2f}")
                    o2.metric("Draw", f"{r['best_odds'].get('draw', 0):.2f}")
                    o3.metric("Away", f"{r['best_odds'].get('away', 0):.2f}")

                if has_bets:
                    st.markdown("**Value Bets Detected**")
                    for b in r["value_bets"]:
                        _icon = {"FUERTE": "🟢", "MODERADO": "🟡", "DÉBIL": "🔴"}.get(b.get("edge_label", ""), "⚪")
                        b1, b2, b3, b4, b5 = st.columns([2, 1, 1, 1, 2])
                        b1.markdown(f"🎯 **{b['outcome']}** {_icon} `{b.get('edge_label','')}`")
                        b2.markdown(f"Odds: **{b['best_odds']:.2f}**")
                        b3.markdown(f"Modelo DC: **{b['model_prob']:.1%}**")
                        b4.markdown(f"Mkt limpio: **{b.get('clean_prob', b['implied_prob']):.1%}**")
                        b5.markdown(f"Edge: **+{b['edge']:.1%}** | EV: **{b.get('ev', 0):+.1%}**")
                        if b.get("kelly"):
                            st.caption(f"Kelly fraccionado: **{b['kelly']:.2%}** del bankroll")
                        st.progress(min(int(b["edge"] * 100), 100))
                elif not r["has_odds"]:
                    st.caption("⚠️ No hay suficiente información para el análisis de este partido.")

            st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("⚠️ Solo con fines educativos. Las apuestas conllevan riesgo de pérdida.")
