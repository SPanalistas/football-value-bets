"""
Football Value Bet Analyzer
Uses football-data.org for fixtures/team stats + The Odds API for odds.
Poisson model with form adjustment (last 5 matches).
"""

import os
import requests
import datetime
from datetime import date
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIG ───────────────────────────────────────────────────────────────────

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_FOOTBALL_DATA_KEY")
ODDS_API_KEY          = os.getenv("ODDS_API_KEY",          "YOUR_ODDS_API_KEY")

FD_BASE   = "https://api.football-data.org/v4"
ODDS_BASE = "https://api.the-odds-api.com/v4"

# football-data.org competition IDs
FD_COMPETITIONS = {
    2021: "Premier League",
    2014: "La Liga",
    2019: "Serie A",
    2002: "Bundesliga",
    2015: "Ligue 1",
    2001: "Champions League",
    2146: "Europa League",
    2000: "World Cup",
}
FD_WC_ID = 2000

# Keep LEAGUES for LEAGUE_FLAG lookup in app.py
LEAGUES = {v: v for v in FD_COMPETITIONS.values()}

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_fifa_world_cup",
]

VALUE_THRESHOLD       = 0.04
RHO                   = -0.13   # Dixon-Coles correlation parameter
MIN_GAMES_HOME        = 0
MIN_GAMES_AWAY        = 0
MIN_GAMES_INTL        = 0
LEAGUE_AVG_HOME_GOALS = 1.50
LEAGUE_AVG_AWAY_GOALS = 1.20
INTL_AVG_GOALS        = 1.15
FORM_WEIGHT           = 0.35
WORLD_CUP_LEAGUE_ID   = FD_WC_ID


def current_season() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


# ─── FOOTBALL-DATA.ORG CLIENT ─────────────────────────────────────────────────

class FootballDataClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})
        self._matches_cache: dict = {}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{FD_BASE}/{endpoint}"
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                print("⚠️  football-data.org rate limit reached.")
                return {}
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            print(f"⚠️  football-data.org error: {e}")
            return {}

    def get_today_matches(self) -> list[dict]:
        """All matches today across tracked competitions.
        API returns empty when dateFrom==dateTo, so we query today→tomorrow and filter."""
        today    = date.today()
        tomorrow = (today + datetime.timedelta(days=1)).isoformat()
        today_s  = today.isoformat()
        data     = self._get("matches", {"dateFrom": today_s, "dateTo": tomorrow})
        all_m    = data.get("matches", [])
        return [
            m for m in all_m
            if m.get("utcDate", "").startswith(today_s)
            and m.get("competition", {}).get("id") in FD_COMPETITIONS
        ]

    def get_team_recent_matches(self, team_id: int, lookback_days: int = 730) -> list[dict]:
        """Up to 38 finished matches for a team across last 2 seasons (stats + form)."""
        if team_id in self._matches_cache:
            return self._matches_cache[team_id]
        today     = date.today()
        date_from = (today - datetime.timedelta(days=lookback_days)).isoformat()
        data      = self._get(f"teams/{team_id}/matches", {
            "dateFrom": date_from,
            "dateTo":   today.isoformat(),
            "status":   "FINISHED",
        })
        matches = data.get("matches", [])
        result  = matches[-38:] if len(matches) > 38 else matches
        self._matches_cache[team_id] = result
        return result


# ─── STATS & FORM FROM RECENT MATCHES ────────────────────────────────────────

def compute_team_stats_from_matches(recent_matches: list, team_id: int) -> dict:
    """
    Build a stats dict (compatible with expected_goals) from recent match results.
    Computes separate home/away attack and defense averages.
    """
    home_scored, home_conceded = [], []
    away_scored, away_conceded = [], []

    for m in recent_matches:
        ht_id = m.get("homeTeam", {}).get("id")
        ft    = m.get("score", {}).get("fullTime", {})
        hg    = ft.get("home") or 0
        ag    = ft.get("away") or 0
        if ht_id == team_id:
            home_scored.append(hg);   home_conceded.append(ag)
        else:
            away_scored.append(ag);   away_conceded.append(hg)

    def avg(lst, fallback):
        return round(sum(lst) / len(lst), 2) if lst else fallback

    h_s = avg(home_scored,   LEAGUE_AVG_HOME_GOALS)
    h_c = avg(home_conceded, LEAGUE_AVG_AWAY_GOALS)
    a_s = avg(away_scored,   LEAGUE_AVG_AWAY_GOALS)
    a_c = avg(away_conceded, LEAGUE_AVG_HOME_GOALS)

    all_scored    = home_scored + away_scored
    all_conceded  = home_conceded + away_conceded
    tot_s = avg(all_scored,   INTL_AVG_GOALS)
    tot_c = avg(all_conceded, INTL_AVG_GOALS)

    return {
        "fixtures": {"played": {
            "home":  len(home_scored),
            "away":  len(away_scored),
            "total": len(home_scored) + len(away_scored),
        }},
        "goals": {
            "for":     {"average": {"home": str(h_s), "away": str(a_s), "total": str(tot_s)}},
            "against": {"average": {"home": str(h_c), "away": str(a_c), "total": str(tot_c)}},
        },
    }


def compute_form_from_fd_matches(recent_matches: list, team_id: int) -> dict:
    """Compute form stats from football-data.org match objects (last 5)."""
    goals_for, goals_against, results = [], [], []
    for m in recent_matches[-5:]:
        ht_id  = m.get("homeTeam", {}).get("id")
        ft     = m.get("score", {}).get("fullTime", {})
        hg     = ft.get("home") or 0
        ag     = ft.get("away") or 0
        winner = m.get("score", {}).get("winner")  # "HOME_TEAM" | "AWAY_TEAM" | "DRAW"
        if ht_id == team_id:
            goals_for.append(hg);  goals_against.append(ag)
            results.append("W" if winner == "HOME_TEAM" else ("D" if winner == "DRAW" else "L"))
        else:
            goals_for.append(ag);  goals_against.append(hg)
            results.append("W" if winner == "AWAY_TEAM" else ("D" if winner == "DRAW" else "L"))
    n = len(goals_for)
    if n == 0:
        return {}
    return {
        "avg_scored":   round(sum(goals_for) / n, 2),
        "avg_conceded": round(sum(goals_against) / n, 2),
        "results":      results,
        "n":            n,
    }


# ─── POISSON MODEL ────────────────────────────────────────────────────────────

def _safe_float(val, fallback: float) -> float:
    try:
        return float(val) if val else fallback
    except (TypeError, ValueError):
        return fallback


def expected_goals(
    home_stats:  dict,
    away_stats:  dict,
    home_form:   dict | None = None,
    away_form:   dict | None = None,
    home_injury: dict | None = None,
    away_injury: dict | None = None,
    is_neutral:  bool = False,
) -> tuple[float, float]:
    avg_h = INTL_AVG_GOALS if is_neutral else LEAGUE_AVG_HOME_GOALS
    avg_a = INTL_AVG_GOALS if is_neutral else LEAGUE_AVG_AWAY_GOALS

    hg = home_stats.get("goals", {})
    ag = away_stats.get("goals", {})

    if is_neutral:
        h_scored   = _safe_float(hg.get("for",     {}).get("average", {}).get("total"), avg_h)
        h_conceded = _safe_float(hg.get("against", {}).get("average", {}).get("total"), avg_a)
        a_scored   = _safe_float(ag.get("for",     {}).get("average", {}).get("total"), avg_a)
        a_conceded = _safe_float(ag.get("against", {}).get("average", {}).get("total"), avg_h)
    else:
        h_scored   = _safe_float(hg.get("for",     {}).get("average", {}).get("home"), avg_h)
        h_conceded = _safe_float(hg.get("against", {}).get("average", {}).get("home"), avg_a)
        a_scored   = _safe_float(ag.get("for",     {}).get("average", {}).get("away"), avg_a)
        a_conceded = _safe_float(ag.get("against", {}).get("average", {}).get("away"), avg_h)

    if home_form and home_form.get("avg_scored") is not None:
        h_scored   = (1 - FORM_WEIGHT) * h_scored   + FORM_WEIGHT * home_form["avg_scored"]
        h_conceded = (1 - FORM_WEIGHT) * h_conceded + FORM_WEIGHT * home_form["avg_conceded"]
    if away_form and away_form.get("avg_scored") is not None:
        a_scored   = (1 - FORM_WEIGHT) * a_scored   + FORM_WEIGHT * away_form["avg_scored"]
        a_conceded = (1 - FORM_WEIGHT) * a_conceded + FORM_WEIGHT * away_form["avg_conceded"]

    h_attack  = h_scored   / avg_h
    h_defense = h_conceded / avg_a
    a_attack  = a_scored   / avg_a
    a_defense = a_conceded / avg_h

    lam_h = max(h_attack * a_defense * avg_h, 0.1)
    lam_a = max(a_attack * h_defense * avg_a, 0.1)

    if home_injury:
        lam_h *= home_injury.get("attack_factor", 1.0)
        lam_a *= home_injury.get("defense_factor", 1.0)
    if away_injury:
        lam_a *= away_injury.get("attack_factor", 1.0)
        lam_h *= away_injury.get("defense_factor", 1.0)

    return round(lam_h, 3), round(lam_a, 3)


def _tau(i: int, j: int, lam_h: float, lam_a: float, rho: float = RHO) -> float:
    """Dixon-Coles correction factor for low-score cells."""
    if i == 0 and j == 0: return 1 - lam_h * lam_a * rho
    if i == 1 and j == 0: return 1 + lam_a * rho
    if i == 0 and j == 1: return 1 + lam_h * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def match_probabilities(lam_h: float, lam_a: float, max_goals: int = 8) -> dict[str, float]:
    p_home_win = p_draw = p_away_win = 0.0
    p_over25 = p_btts = 0.0
    p_home_s = p_draw_s = p_away_s = 0.0
    dc_probs: dict = {}

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_pois = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
            p_dc   = max(_tau(h, a, lam_h, lam_a) * p_pois, 0.0)
            dc_probs[(h, a)] = p_dc
            if h > a:    p_home_s += p_pois
            elif h == a: p_draw_s += p_pois
            else:        p_away_s += p_pois

    total_dc = sum(dc_probs.values())
    if abs(total_dc - 1.0) > 0.001 and total_dc > 0:
        dc_probs = {k: v / total_dc for k, v in dc_probs.items()}

    for (h, a), p in dc_probs.items():
        if h > a:           p_home_win += p
        elif h == a:        p_draw     += p
        else:               p_away_win += p
        if h + a > 2.5:     p_over25   += p
        if h > 0 and a > 0: p_btts     += p

    return {
        "home_win":          p_home_win,
        "draw":              p_draw,
        "away_win":          p_away_win,
        "over_2_5":          p_over25,
        "btts":              p_btts,
        "poisson_home_win":  p_home_s,
        "poisson_draw":      p_draw_s,
        "poisson_away_win":  p_away_s,
    }


# ─── ODDS CLIENT ─────────────────────────────────────────────────────────────

class OddsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_odds(self, sport_key: str) -> list[dict]:
        url = f"{ODDS_BASE}/sports/{sport_key}/odds"
        try:
            r = self.session.get(url, params={
                "apiKey":     self.api_key,
                "regions":    "eu",
                "markets":    "h2h",
                "oddsFormat": "decimal",
            }, timeout=15)
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining", "?")
            print(f"  ✅  {sport_key}: {len(r.json())} events | Requests left: {remaining}")
            return r.json()
        except requests.HTTPError as e:
            print(f"  ⚠️  Odds unavailable for {sport_key}: {e}")
            return []

    def best_odds(self, event: dict) -> dict[str, float]:
        best = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name  = outcome.get("name", "")
                    price = outcome.get("price", 0.0)
                    if name == event.get("home_team"):
                        best["home"] = max(best["home"], price)
                    elif name == event.get("away_team"):
                        best["away"] = max(best["away"], price)
                    elif name == "Draw":
                        best["draw"] = max(best["draw"], price)
        return best


# ─── VALUE BET DETECTOR ───────────────────────────────────────────────────────

def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds > 1 else 0.0


def remove_overround(odds_h: float, odds_d: float, odds_a: float) -> dict[str, float]:
    """Return fair probabilities stripping bookmaker margin (Pinnacle-style normalization)."""
    inv = {"home": 1/odds_h if odds_h > 1 else 0,
           "draw": 1/odds_d if odds_d > 1 else 0,
           "away": 1/odds_a if odds_a > 1 else 0}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()} if total else {"home": 0, "draw": 0, "away": 0}


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    if odds <= 1 or prob <= 0:
        return 0.0
    f = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
    return round(max(0.0, f * fraction), 4)


def detect_value_bets(model_probs: dict, best_odds: dict, threshold: float = VALUE_THRESHOLD) -> list[dict]:
    clean = remove_overround(
        best_odds.get("home", 0),
        best_odds.get("draw", 0),
        best_odds.get("away", 0),
    )
    mapping = [
        ("home_win", "home", "Home Win"),
        ("draw",     "draw", "Draw"),
        ("away_win", "away", "Away Win"),
    ]
    bets = []
    for prob_key, odds_key, label in mapping:
        prob       = model_probs.get(prob_key, 0)
        odds       = best_odds.get(odds_key, 0)
        clean_prob = clean.get(odds_key, 0)
        if odds <= 1:
            continue
        edge = prob - clean_prob                      # vs fair market probability
        ev   = prob * (odds - 1) - (1 - prob)        # expected value per unit staked
        if edge >= threshold and ev > 0:
            frac       = 0.50 if edge >= 0.07 else 0.25
            kelly      = kelly_fraction(prob, odds, frac)
            edge_label = ("FUERTE"   if edge >= 0.07 and ev >= 0.10
                          else "MODERADO" if edge >= 0.04
                          else "DÉBIL")
            bets.append({
                "outcome":      label,
                "model_prob":   prob,
                "implied_prob": implied_probability(odds),
                "clean_prob":   clean_prob,
                "best_odds":    odds,
                "edge":         edge,
                "ev":           ev,
                "kelly":        kelly,
                "edge_label":   edge_label,
            })
    return sorted(bets, key=lambda x: x["edge"], reverse=True)


# ─── NAME MATCHING ────────────────────────────────────────────────────────────

_STRIP_WORDS = {"fc", "cf", "ac", "sc", "rc", "if", "bk", "sk", "fk", "sv",
                "club", "de", "del", "la", "el", "los", "the", "united",
                "city", "town", "athletic", "athletics", "sport", "sporting"}

def normalize(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = "".join(c if c.isalnum() or c == " " else " " for c in name)
    tokens = [t for t in name.split() if t not in _STRIP_WORDS]
    return " ".join(tokens)


def teams_match(fd_home: str, fd_away: str, odds_home: str, odds_away: str) -> bool:
    def similar(a: str, b: str) -> bool:
        a, b = normalize(a), normalize(b)
        if a == b: return True
        if a in b or b in a: return True
        ta, tb = set(a.split()), set(b.split())
        shorter = ta if len(ta) <= len(tb) else tb
        if shorter and len(ta & tb) >= max(1, len(shorter) - 1): return True
        return len(a) >= 4 and len(b) >= 4 and a[:5] == b[:5]
    return similar(fd_home, odds_home) and similar(fd_away, odds_away)


# ─── DEMO DATA ────────────────────────────────────────────────────────────────

def _fd_match(match_id, comp_id, home_id, home_name, away_id, away_name, time="15:00"):
    return {
        "id":          match_id,
        "utcDate":     f"{date.today().isoformat()}T{time}:00Z",
        "status":      "SCHEDULED",
        "competition": {"id": comp_id, "name": FD_COMPETITIONS[comp_id]},
        "homeTeam":    {"id": home_id, "name": home_name},
        "awayTeam":    {"id": away_id, "name": away_name},
        "score":       {"winner": None, "fullTime": {"home": None, "away": None}},
    }

def _fd_result(home_id, home_name, away_id, away_name, hg, ag):
    winner = "HOME_TEAM" if hg > ag else ("AWAY_TEAM" if ag > hg else "DRAW")
    return {
        "homeTeam": {"id": home_id, "name": home_name},
        "awayTeam": {"id": away_id, "name": away_name},
        "score": {"winner": winner, "fullTime": {"home": hg, "away": ag}},
    }

DEMO_MATCHES = [
    _fd_match(1, 2021, 57,  "Arsenal",          63,  "Manchester City",   "15:00"),
    _fd_match(2, 2021, 61,  "Chelsea",           73,  "Tottenham",         "17:30"),
    _fd_match(3, 2021, 64,  "Liverpool",         67,  "Manchester Utd",    "20:00"),
    _fd_match(4, 2014, 86,  "Real Madrid",       81,  "Barcelona",         "20:00"),
    _fd_match(5, 2014, 78,  "Atletico Madrid",   559, "Sevilla",           "18:00"),
    _fd_match(6, 2019, 108, "Inter Milan",       98,  "AC Milan",          "20:45"),
    _fd_match(7, 2019, 109, "Juventus",          113, "Napoli",            "18:00"),
    _fd_match(8, 2002, 5,   "Bayern Munich",     4,   "Borussia Dortmund", "18:30"),
    _fd_match(9, 2002, 3,   "Bayer Leverkusen",  721, "RB Leipzig",        "15:30"),
    _fd_match(10,2015, 524, "PSG",               548, "Monaco",            "20:45"),
    _fd_match(11,2015, 516, "Lyon",              516, "Marseille",         "17:00"),
    _fd_match(12,2001, 86,  "Real Madrid",       108, "Inter Milan",       "21:00"),
    _fd_match(13,2146, 674, "Ajax",              100, "Roma",              "21:00"),
    _fd_match(14,2000, 759, "Brazil",            762, "Argentina",         "18:00"),
    _fd_match(15,2000, 760, "France",            758, "Germany",           "21:00"),
    _fd_match(16,2000, 760, "Spain",             770, "England",           "15:00"),
    _fd_match(17,2000, 765, "Portugal",          772, "Morocco",           "20:00"),
]

def _recent(home_id, home_name, away_id, away_name, results):
    matches = []
    for r in results:
        hg, ag = r
        matches.append(_fd_result(home_id, home_name, away_id, away_name, hg, ag))
    return matches

DEMO_RECENT = {
    57:  [_fd_result(57,"Arsenal",  63,"Man City",2,0), _fd_result(73,"Tottenham",57,"Arsenal",0,2),
          _fd_result(57,"Arsenal",  61,"Chelsea", 3,1), _fd_result(64,"Liverpool",57,"Arsenal",1,2),
          _fd_result(57,"Arsenal",  67,"Man Utd", 2,0), _fd_result(57,"Arsenal",  5,"Bayern",  1,3),
          _fd_result(57,"Arsenal",  98,"AC Milan",2,1), _fd_result(61,"Chelsea",  57,"Arsenal",0,1),
          _fd_result(57,"Arsenal",  73,"Tottenham",1,0),_fd_result(67,"Man Utd",  57,"Arsenal",0,2)],
    63:  [_fd_result(63,"Man City", 64,"Liverpool",1,2),_fd_result(57,"Arsenal",  63,"Man City",0,2),
          _fd_result(63,"Man City", 73,"Tottenham",3,0),_fd_result(61,"Chelsea",  63,"Man City",1,1),
          _fd_result(63,"Man City", 67,"Man Utd",  2,0),_fd_result(63,"Man City", 5,"Bayern",   1,2),
          _fd_result(63,"Man City", 57,"Arsenal",  2,1),_fd_result(64,"Liverpool",63,"Man City",1,1),
          _fd_result(63,"Man City", 61,"Chelsea",  4,0),_fd_result(73,"Tottenham",63,"Man City",0,3)],
    61:  [_fd_result(61,"Chelsea",  63,"Man City",1,1),_fd_result(57,"Arsenal",  61,"Chelsea",3,1),
          _fd_result(61,"Chelsea",  67,"Man Utd", 2,0),_fd_result(73,"Tottenham",61,"Chelsea",1,2),
          _fd_result(61,"Chelsea",  64,"Liverpool",0,2)],
    73:  [_fd_result(73,"Tottenham",63,"Man City",0,3),_fd_result(57,"Arsenal",73,"Tottenham",1,0),
          _fd_result(73,"Tottenham",64,"Liverpool",1,2),_fd_result(61,"Chelsea",73,"Tottenham",2,1),
          _fd_result(73,"Tottenham",67,"Man Utd",  2,1)],
    64:  [_fd_result(64,"Liverpool",63,"Man City",2,1),_fd_result(57,"Arsenal",64,"Liverpool",2,1),
          _fd_result(64,"Liverpool",61,"Chelsea", 3,0),_fd_result(64,"Liverpool",73,"Tottenham",2,1),
          _fd_result(67,"Man Utd",  64,"Liverpool",0,3)],
    67:  [_fd_result(57,"Arsenal",67,"Man Utd",2,0),_fd_result(67,"Man Utd",61,"Chelsea",0,1),
          _fd_result(73,"Tottenham",67,"Man Utd",2,1),_fd_result(67,"Man Utd",64,"Liverpool",0,3),
          _fd_result(67,"Man Utd",63,"Man City",0,2)],
    86:  [_fd_result(86,"Real Madrid",81,"Barcelona",2,1),_fd_result(86,"Real Madrid",78,"Atletico",1,0),
          _fd_result(559,"Sevilla",86,"Real Madrid",0,3),_fd_result(86,"Real Madrid",108,"Inter",2,0),
          _fd_result(81,"Barcelona",86,"Real Madrid",1,2)],
    81:  [_fd_result(81,"Barcelona",86,"Real Madrid",1,2),_fd_result(81,"Barcelona",78,"Atletico",3,0),
          _fd_result(559,"Sevilla",81,"Barcelona",0,2),_fd_result(81,"Barcelona",108,"Inter",2,1),
          _fd_result(86,"Real Madrid",81,"Barcelona",2,1)],
    78:  [_fd_result(78,"Atletico",559,"Sevilla",1,0),_fd_result(86,"Real Madrid",78,"Atletico",1,0),
          _fd_result(78,"Atletico",81,"Barcelona",0,3),_fd_result(78,"Atletico",86,"Real Madrid",0,1),
          _fd_result(559,"Sevilla",78,"Atletico",0,1)],
    559: [_fd_result(559,"Sevilla",86,"Real Madrid",0,3),_fd_result(81,"Barcelona",559,"Sevilla",2,0),
          _fd_result(559,"Sevilla",78,"Atletico",0,1),_fd_result(86,"Real Madrid",559,"Sevilla",3,0),
          _fd_result(559,"Sevilla",81,"Barcelona",0,2)],
    108: [_fd_result(108,"Inter",98,"AC Milan",2,1),_fd_result(108,"Inter",109,"Juventus",2,0),
          _fd_result(113,"Napoli",108,"Inter",0,2),_fd_result(108,"Inter",86,"Real Madrid",0,2),
          _fd_result(98,"AC Milan",108,"Inter",1,2)],
    98:  [_fd_result(98,"AC Milan",108,"Inter",1,2),_fd_result(98,"AC Milan",109,"Juventus",1,1),
          _fd_result(113,"Napoli",98,"AC Milan",2,1),_fd_result(108,"Inter",98,"AC Milan",2,1),
          _fd_result(98,"AC Milan",113,"Napoli",2,0)],
    109: [_fd_result(109,"Juventus",113,"Napoli",1,1),_fd_result(108,"Inter",109,"Juventus",2,0),
          _fd_result(109,"Juventus",98,"AC Milan",1,1),_fd_result(113,"Napoli",109,"Juventus",1,0),
          _fd_result(109,"Juventus",108,"Inter",0,2)],
    113: [_fd_result(113,"Napoli",108,"Inter",0,2),_fd_result(113,"Napoli",109,"Juventus",1,0),
          _fd_result(98,"AC Milan",113,"Napoli",2,0),_fd_result(113,"Napoli",98,"AC Milan",3,1),
          _fd_result(109,"Juventus",113,"Napoli",1,1)],
    5:   [_fd_result(5,"Bayern",4,"Dortmund",3,1),_fd_result(5,"Bayern",3,"Leverkusen",2,1),
          _fd_result(721,"RB Leipzig",5,"Bayern",0,4),_fd_result(5,"Bayern",57,"Arsenal",3,1),
          _fd_result(4,"Dortmund",5,"Bayern",1,3)],
    4:   [_fd_result(4,"Dortmund",721,"RB Leipzig",2,1),_fd_result(5,"Bayern",4,"Dortmund",3,1),
          _fd_result(4,"Dortmund",3,"Leverkusen",1,2),_fd_result(4,"Dortmund",5,"Bayern",1,3),
          _fd_result(721,"RB Leipzig",4,"Dortmund",1,2)],
    3:   [_fd_result(3,"Leverkusen",721,"RB Leipzig",2,1),_fd_result(5,"Bayern",3,"Leverkusen",2,1),
          _fd_result(3,"Leverkusen",4,"Dortmund",2,1),_fd_result(721,"RB Leipzig",3,"Leverkusen",0,2),
          _fd_result(3,"Leverkusen",5,"Bayern",1,2)],
    721: [_fd_result(721,"RB Leipzig",5,"Bayern",0,4),_fd_result(4,"Dortmund",721,"RB Leipzig",2,1),
          _fd_result(721,"RB Leipzig",3,"Leverkusen",0,2),_fd_result(3,"Leverkusen",721,"RB Leipzig",2,1),
          _fd_result(721,"RB Leipzig",4,"Dortmund",1,2)],
    524: [_fd_result(524,"PSG",548,"Monaco",3,0),_fd_result(524,"PSG",516,"Lyon",4,0),
          _fd_result(516,"Marseille",524,"PSG",0,2),_fd_result(524,"PSG",548,"Monaco",2,0),
          _fd_result(548,"Monaco",524,"PSG",1,3)],
    548: [_fd_result(524,"PSG",548,"Monaco",3,0),_fd_result(548,"Monaco",516,"Lyon",2,1),
          _fd_result(516,"Marseille",548,"Monaco",1,2),_fd_result(548,"Monaco",524,"PSG",1,3),
          _fd_result(548,"Monaco",516,"Marseille",2,0)],
    516: [_fd_result(516,"Lyon",548,"Monaco",0,2),_fd_result(524,"PSG",516,"Lyon",4,0),
          _fd_result(516,"Lyon",516,"Marseille",1,1),_fd_result(516,"Marseille",516,"Lyon",2,0),
          _fd_result(516,"Lyon",524,"PSG",0,4)],
    674: [_fd_result(674,"Ajax",100,"Roma",2,1),_fd_result(674,"Ajax",57,"Arsenal",1,2),
          _fd_result(100,"Roma",674,"Ajax",0,2),_fd_result(674,"Ajax",4,"Dortmund",3,1),
          _fd_result(674,"Ajax",100,"Roma",3,0)],
    100: [_fd_result(100,"Roma",674,"Ajax",0,2),_fd_result(100,"Roma",108,"Inter",1,2),
          _fd_result(674,"Ajax",100,"Roma",2,1),_fd_result(100,"Roma",109,"Juventus",2,1),
          _fd_result(100,"Roma",674,"Ajax",0,3)],
    759: [_fd_result(759,"Brazil",762,"Argentina",1,1),_fd_result(759,"Brazil",760,"France",2,1),
          _fd_result(759,"Brazil",770,"England",3,0),_fd_result(762,"Argentina",759,"Brazil",0,2),
          _fd_result(759,"Brazil",765,"Portugal",2,0)],
    762: [_fd_result(762,"Argentina",759,"Brazil",0,2),_fd_result(762,"Argentina",760,"France",2,0),
          _fd_result(762,"Argentina",770,"England",3,1),_fd_result(759,"Brazil",762,"Argentina",2,0),
          _fd_result(762,"Argentina",765,"Portugal",1,0)],
    760: [_fd_result(760,"France",758,"Germany",2,0),_fd_result(759,"Brazil",760,"France",2,1),
          _fd_result(760,"France",770,"England",2,1),_fd_result(762,"Argentina",760,"France",2,0),
          _fd_result(760,"France",772,"Morocco",3,0)],
    758: [_fd_result(758,"Germany",760,"France",0,2),_fd_result(758,"Germany",770,"England",2,1),
          _fd_result(758,"Germany",765,"Portugal",2,2),_fd_result(760,"France",758,"Germany",2,0),
          _fd_result(758,"Germany",762,"Argentina",1,2)],
    765: [_fd_result(765,"Portugal",772,"Morocco",3,0),_fd_result(765,"Portugal",770,"England",2,1),
          _fd_result(759,"Brazil",765,"Portugal",2,0),_fd_result(762,"Argentina",765,"Portugal",1,0),
          _fd_result(765,"Portugal",760,"France",1,2)],
    770: [_fd_result(770,"England",758,"Germany",1,2),_fd_result(760,"France",770,"England",2,1),
          _fd_result(770,"England",772,"Morocco",2,0),_fd_result(762,"Argentina",770,"England",3,1),
          _fd_result(765,"Portugal",770,"England",2,1)],
    772: [_fd_result(772,"Morocco",765,"Portugal",0,3),_fd_result(770,"England",772,"Morocco",2,0),
          _fd_result(772,"Morocco",760,"France",0,3),_fd_result(772,"Morocco",762,"Argentina",0,2),
          _fd_result(772,"Morocco",758,"Germany",1,2)],
}

_NO_INJ = {"attack_factor": 1.0, "defense_factor": 1.0, "players": []}

def demo_injury(team_id: int) -> dict:
    return _NO_INJ

def _odds_event(home_name, away_name, home_price, draw_price, away_price):
    return {
        "home_team": home_name,
        "away_team": away_name,
        "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
            {"name": home_name,  "price": home_price},
            {"name": "Draw",     "price": draw_price},
            {"name": away_name,  "price": away_price},
        ]}]}],
    }

DEMO_ODDS = [
    _odds_event("Arsenal",          "Manchester City",   2.10, 3.40, 3.50),
    _odds_event("Chelsea",          "Tottenham",         2.20, 3.30, 3.20),
    _odds_event("Liverpool",        "Manchester Utd",    1.60, 3.80, 5.50),
    _odds_event("Real Madrid",      "Barcelona",         2.30, 3.40, 3.00),
    _odds_event("Atletico Madrid",  "Sevilla",           1.90, 3.50, 4.00),
    _odds_event("Inter Milan",      "AC Milan",          2.00, 3.30, 3.80),
    _odds_event("Juventus",         "Napoli",            2.40, 3.20, 3.00),
    _odds_event("Bayern Munich",    "Borussia Dortmund", 1.60, 4.00, 5.50),
    _odds_event("Bayer Leverkusen", "RB Leipzig",        2.10, 3.30, 3.50),
    _odds_event("PSG",              "Monaco",            1.40, 4.50, 7.00),
    _odds_event("Lyon",             "Marseille",         2.50, 3.20, 2.80),
    _odds_event("Real Madrid",      "Inter Milan",       2.10, 3.50, 3.40),
    _odds_event("Ajax",             "Roma",              2.20, 3.30, 3.20),
    _odds_event("Brazil",           "Argentina",         2.50, 3.20, 2.90),
    _odds_event("France",           "Germany",           2.10, 3.40, 3.50),
    _odds_event("Spain",            "England",           2.30, 3.30, 3.10),
    _odds_event("Portugal",         "Morocco",           1.75, 3.60, 4.80),
]
