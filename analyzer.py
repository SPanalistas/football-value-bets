"""
Football Value Bet Analyzer
Poisson model + API-Football stats + The Odds API.
Includes: form adjustment (last 5), injury/suspension penalties, World Cup neutral-venue path.
"""

import os
import requests
from datetime import date
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "YOUR_API_FOOTBALL_KEY")
ODDS_API_KEY     = os.getenv("ODDS_API_KEY",     "YOUR_ODDS_API_KEY")

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_BASE     = "https://api.the-odds-api.com/v4"

LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    135: "Serie A",
    78:  "Bundesliga",
    61:  "Ligue 1",
    2:   "Champions League",
    3:   "Europa League",
    1:   "World Cup",
}

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

# Club league settings
VALUE_THRESHOLD       = 0.05
MIN_GAMES_HOME        = 3
MIN_GAMES_AWAY        = 3
LEAGUE_AVG_HOME_GOALS = 1.50
LEAGUE_AVG_AWAY_GOALS = 1.20

# World Cup / international settings
WORLD_CUP_LEAGUE_ID = 1
WORLD_CUP_SEASON    = 2026     # update every 4 years
INTL_AVG_GOALS      = 1.15    # neutral venue — no home advantage
MIN_GAMES_INTL      = 3       # minimum qualifier games

# Form model
FORM_MATCHES = 5              # recent matches to consider
FORM_WEIGHT  = 0.35           # 35% recent form, 65% season average


def current_season() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


# ─── API-FOOTBALL CLIENT ──────────────────────────────────────────────────────

class APIFootballClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": api_key})
        self._stats_cache:  dict = {}
        self._form_cache:   dict = {}
        self._injury_cache: dict = {}

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{API_FOOTBALL_BASE}/{endpoint}"
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                print("⚠️  API-Football rate limit reached.")
                return {}
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            print(f"⚠️  API-Football error: {e}")
            return {}

    def get_today_fixtures(self, league_id: int, season: int) -> list[dict]:
        data = self._get("fixtures", {
            "date":   date.today().isoformat(),
            "league": league_id,
            "season": season,
        })
        return data.get("response", [])

    def get_team_stats(self, team_id: int, league_id: int, season: int) -> dict:
        key = (team_id, league_id, season)
        if key not in self._stats_cache:
            data = self._get("teams/statistics", {
                "team":   team_id,
                "league": league_id,
                "season": season,
            })
            self._stats_cache[key] = data.get("response", {})
        return self._stats_cache[key]

    def get_recent_fixtures(self, team_id: int, n: int = FORM_MATCHES) -> list[dict]:
        """Last N finished fixtures across all competitions (for form)."""
        if team_id not in self._form_cache:
            data = self._get("fixtures", {
                "team":   team_id,
                "last":   n,
                "status": "FT",
            })
            self._form_cache[team_id] = data.get("response", [])
        return self._form_cache[team_id]

    def get_injuries(self, team_id: int, season: int) -> list[dict]:
        """Current injuries and suspensions for a team."""
        key = (team_id, season)
        if key not in self._injury_cache:
            data = self._get("injuries", {"team": team_id, "season": season})
            self._injury_cache[key] = data.get("response", [])
        return self._injury_cache[key]


# ─── FORM & INJURY HELPERS ────────────────────────────────────────────────────

def compute_form_stats(fixtures: list, team_id: int) -> dict:
    """Derive recent attack/defense averages from last N finished fixtures."""
    if not fixtures:
        return {}
    goals_for, goals_against, results = [], [], []
    for f in fixtures:
        home_id = f.get("teams", {}).get("home", {}).get("id")
        g       = f.get("goals", {})
        hg      = g.get("home") or 0
        ag      = g.get("away") or 0
        winner  = f.get("teams", {}).get("home", {}).get("winner")  # True/False/None
        if home_id == team_id:
            goals_for.append(hg); goals_against.append(ag)
            results.append("W" if winner is True else ("D" if winner is None else "L"))
        else:
            goals_for.append(ag); goals_against.append(hg)
            results.append("L" if winner is True else ("D" if winner is None else "W"))
    n = len(goals_for)
    if n == 0:
        return {}
    return {
        "avg_scored":   round(sum(goals_for) / n, 2),
        "avg_conceded": round(sum(goals_against) / n, 2),
        "results":      results,
        "n":            n,
    }


def compute_injury_factor(injuries: list) -> dict:
    """Return attack/defense multipliers based on current injuries/suspensions."""
    ATK = {"Attacker": 0.07, "Midfielder": 0.03, "Defender": 0.01, "Goalkeeper": 0.02}
    DEF = {"Attacker": 0.01, "Midfielder": 0.03, "Defender": 0.06, "Goalkeeper": 0.07}
    atk_pen = def_pen = 0.0
    players = []
    for inj in injuries:
        player = inj.get("player", {})
        pos    = player.get("type", "")
        reason = inj.get("injury", {}).get("type", "")
        if reason in ("Injured", "Suspended"):
            atk_pen += ATK.get(pos, 0.02)
            def_pen += DEF.get(pos, 0.02)
            players.append({
                "name":     player.get("name", "Unknown"),
                "position": pos,
                "reason":   reason,
            })
    return {
        "attack_factor":  max(round(1.0 - min(atk_pen, 0.25), 3), 0.75),
        "defense_factor": min(round(1.0 + min(def_pen, 0.20), 3), 1.20),
        "players":        players,
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
    """
    Compute λ_home / λ_away using:
      - Season attack/defense ratios (home/away for clubs; total for World Cup)
      - Weighted blend with recent form (FORM_WEIGHT = 35%)
      - Per-position injury/suspension penalty
    """
    avg_h = INTL_AVG_GOALS if is_neutral else LEAGUE_AVG_HOME_GOALS
    avg_a = INTL_AVG_GOALS if is_neutral else LEAGUE_AVG_AWAY_GOALS

    hg = home_stats.get("goals", {})
    ag = away_stats.get("goals", {})
    slot = "total" if is_neutral else None

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

    # Blend season average with recent form
    if home_form and home_form.get("avg_scored") is not None:
        h_scored   = (1 - FORM_WEIGHT) * h_scored   + FORM_WEIGHT * home_form["avg_scored"]
        h_conceded = (1 - FORM_WEIGHT) * h_conceded + FORM_WEIGHT * home_form["avg_conceded"]
    if away_form and away_form.get("avg_scored") is not None:
        a_scored   = (1 - FORM_WEIGHT) * a_scored   + FORM_WEIGHT * away_form["avg_scored"]
        a_conceded = (1 - FORM_WEIGHT) * a_conceded + FORM_WEIGHT * away_form["avg_conceded"]

    # Attack/defense strength ratios
    h_attack  = h_scored   / avg_h
    h_defense = h_conceded / avg_a
    a_attack  = a_scored   / avg_a
    a_defense = a_conceded / avg_h

    lam_h = max(h_attack * a_defense * avg_h, 0.1)
    lam_a = max(a_attack * h_defense * avg_a, 0.1)

    # Injury penalties
    if home_injury:
        lam_h *= home_injury["attack_factor"]   # injured home attackers → home scores less
        lam_a *= home_injury["defense_factor"]  # injured home defenders → away scores more
    if away_injury:
        lam_a *= away_injury["attack_factor"]   # injured away attackers → away scores less
        lam_h *= away_injury["defense_factor"]  # injured away defenders → home scores more

    return round(lam_h, 3), round(lam_a, 3)


def match_probabilities(lam_h: float, lam_a: float, max_goals: int = 8) -> dict[str, float]:
    p_home_win = p_draw = p_away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
            if h > a:    p_home_win += p
            elif h == a: p_draw     += p
            else:        p_away_win += p
    total = p_home_win + p_draw + p_away_win
    return {
        "home_win": p_home_win / total,
        "draw":     p_draw     / total,
        "away_win": p_away_win / total,
    }


# ─── ODDS CLIENT ─────────────────────────────────────────────────────────────

class OddsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_odds(self, sport_key: str) -> list[dict]:
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        try:
            r = self.session.get(url, params={
                "apiKey":      self.api_key,
                "regions":     "eu",
                "markets":     "h2h",
                "oddsFormat":  "decimal",
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


def detect_value_bets(model_probs: dict, best_odds: dict, threshold: float = VALUE_THRESHOLD) -> list[dict]:
    mapping = [
        ("home_win", "home", "Home Win"),
        ("draw",     "draw", "Draw"),
        ("away_win", "away", "Away Win"),
    ]
    bets = []
    for prob_key, odds_key, label in mapping:
        prob = model_probs.get(prob_key, 0)
        odds = best_odds.get(odds_key, 0)
        if odds <= 1:
            continue
        ev = prob * odds - 1.0
        if ev >= threshold:
            bets.append({
                "outcome":      label,
                "model_prob":   prob,
                "implied_prob": implied_probability(odds),
                "best_odds":    odds,
                "edge":         ev,
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

def _fixture(fixture_id, league_id, home_id, home_name, away_id, away_name, time="15:00"):
    return {
        "fixture": {"id": fixture_id, "date": f"{date.today().isoformat()}T{time}:00+00:00"},
        "league":  {"id": league_id, "name": LEAGUES[league_id]},
        "teams": {
            "home": {"id": home_id, "name": home_name},
            "away": {"id": away_id, "name": away_name},
        },
    }

def _stats(home_played, away_played, scored_home, scored_away, conceded_home, conceded_away):
    total = home_played + away_played
    s_tot = round((scored_home*home_played + scored_away*away_played)/total, 2) if total else scored_home
    c_tot = round((conceded_home*home_played + conceded_away*away_played)/total, 2) if total else conceded_home
    return {
        "fixtures": {"played": {"home": home_played, "away": away_played, "total": total}},
        "goals": {
            "for":     {"average": {"home": str(scored_home), "away": str(scored_away), "total": str(s_tot)}},
            "against": {"average": {"home": str(conceded_home), "away": str(conceded_away), "total": str(c_tot)}},
        },
    }

DEMO_FIXTURES = {
    39: [
        _fixture(1, 39,  40, "Liverpool",      50, "Manchester City", "15:00"),
        _fixture(2, 39,  42, "Arsenal",        49, "Chelsea",         "17:30"),
        _fixture(3, 39,  33, "Manchester Utd", 47, "Tottenham",       "20:00"),
    ],
    140: [
        _fixture(4,  140, 541, "Real Madrid",     529, "Barcelona",   "20:00"),
        _fixture(5,  140, 530, "Atletico Madrid", 536, "Sevilla",     "18:00"),
    ],
    135: [
        _fixture(6,  135, 505, "Inter Milan", 489, "AC Milan", "20:45"),
        _fixture(7,  135, 496, "Juventus",    492, "Napoli",   "18:00"),
    ],
    78: [
        _fixture(8,  78, 157, "Bayern Munich",    165, "Borussia Dortmund", "18:30"),
        _fixture(9,  78, 168, "Bayer Leverkusen", 173, "RB Leipzig",        "15:30"),
    ],
    61: [
        _fixture(10, 61, 85, "PSG",  91, "Monaco",    "20:45"),
        _fixture(11, 61, 80, "Lyon", 81, "Marseille", "17:00"),
    ],
    2: [_fixture(12, 2, 541, "Real Madrid", 505, "Inter Milan", "21:00")],
    3: [_fixture(13, 3, 194, "Ajax",        497, "Roma",        "21:00")],
    1: [
        _fixture(14, 1,   6, "Brazil",   26,   "Argentina", "18:00"),
        _fixture(15, 1,   2, "France",   25,   "Germany",   "21:00"),
        _fixture(16, 1,   9, "Spain",    10,   "England",   "15:00"),
        _fixture(17, 1,  27, "Portugal", 1580, "Morocco",   "20:00"),
    ],
}

DEMO_TEAM_STATS = {
    # Premier League
    40:  _stats(17, 16, 2.2, 1.5, 0.8, 1.0),  # Liverpool
    50:  _stats(17, 16, 1.9, 1.4, 0.9, 1.0),  # Man City
    42:  _stats(17, 16, 1.8, 1.2, 0.7, 0.9),  # Arsenal
    49:  _stats(17, 16, 1.5, 1.1, 1.2, 1.3),  # Chelsea
    33:  _stats(17, 16, 1.2, 0.9, 1.4, 1.5),  # Man Utd
    47:  _stats(17, 16, 1.6, 1.2, 1.1, 1.3),  # Tottenham
    # La Liga
    541: _stats(18, 17, 2.3, 1.6, 0.7, 0.9),  # Real Madrid
    529: _stats(18, 17, 2.5, 1.7, 0.8, 1.0),  # Barcelona
    530: _stats(18, 17, 1.5, 0.9, 0.7, 0.9),  # Atletico Madrid
    536: _stats(18, 17, 1.2, 0.8, 1.3, 1.4),  # Sevilla
    # Serie A
    505: _stats(18, 17, 2.1, 1.4, 0.6, 0.8),  # Inter Milan
    489: _stats(18, 17, 1.7, 1.1, 1.0, 1.2),  # AC Milan
    496: _stats(18, 17, 1.5, 0.9, 0.9, 1.1),  # Juventus
    492: _stats(18, 17, 1.8, 1.2, 1.0, 1.2),  # Napoli
    # Bundesliga
    157: _stats(17, 16, 2.9, 2.0, 1.0, 1.2),  # Bayern Munich
    165: _stats(17, 16, 2.0, 1.4, 1.4, 1.5),  # Borussia Dortmund
    168: _stats(17, 16, 1.9, 1.3, 0.8, 1.0),  # Bayer Leverkusen
    173: _stats(17, 16, 1.7, 1.1, 1.1, 1.3),  # RB Leipzig
    # Ligue 1
    85:  _stats(18, 17, 2.8, 2.1, 0.6, 0.8),  # PSG
    91:  _stats(18, 17, 1.9, 1.3, 1.1, 1.2),  # Monaco
    80:  _stats(18, 17, 1.4, 1.0, 1.3, 1.4),  # Lyon
    81:  _stats(18, 17, 1.6, 1.1, 1.1, 1.3),  # Marseille
    # UCL / UEL
    194: _stats(6,  6,  2.2, 1.5, 0.8, 1.0),  # Ajax
    497: _stats(6,  6,  1.5, 0.9, 1.0, 1.2),  # Roma
    # World Cup national teams (qualifier campaign averages, total used for neutral venue)
    6:    _stats(5, 5, 2.2, 1.6, 0.6, 0.8),   # Brazil
    26:   _stats(5, 5, 2.5, 1.8, 0.5, 0.7),   # Argentina
    2:    _stats(5, 5, 2.0, 1.5, 0.8, 1.0),   # France
    25:   _stats(5, 5, 1.8, 1.4, 1.0, 1.1),   # Germany
    9:    _stats(5, 5, 2.0, 1.5, 0.7, 0.9),   # Spain
    10:   _stats(5, 5, 1.7, 1.3, 1.0, 1.1),   # England
    27:   _stats(5, 5, 1.9, 1.4, 0.8, 1.0),   # Portugal
    1580: _stats(5, 5, 1.2, 0.8, 0.8, 1.0),   # Morocco
}

DEMO_FORM = {
    40:   {"avg_scored": 2.8, "avg_conceded": 0.6, "results": ["W","W","W","W","D"]},  # Liverpool — on fire
    50:   {"avg_scored": 1.4, "avg_conceded": 1.2, "results": ["D","L","W","D","W"]},  # Man City — shaky
    42:   {"avg_scored": 2.0, "avg_conceded": 0.8, "results": ["W","W","D","W","W"]},  # Arsenal
    49:   {"avg_scored": 1.6, "avg_conceded": 1.4, "results": ["W","D","L","W","D"]},  # Chelsea
    33:   {"avg_scored": 0.8, "avg_conceded": 1.8, "results": ["L","L","D","L","W"]},  # Man Utd — poor
    47:   {"avg_scored": 1.8, "avg_conceded": 1.0, "results": ["W","D","W","L","W"]},  # Tottenham
    541:  {"avg_scored": 2.6, "avg_conceded": 0.8, "results": ["W","W","W","D","W"]},  # Real Madrid
    529:  {"avg_scored": 2.8, "avg_conceded": 1.0, "results": ["W","W","D","W","W"]},  # Barcelona
    530:  {"avg_scored": 1.2, "avg_conceded": 0.6, "results": ["D","W","D","D","W"]},  # Atletico
    536:  {"avg_scored": 1.0, "avg_conceded": 1.6, "results": ["L","D","W","L","D"]},  # Sevilla
    505:  {"avg_scored": 2.4, "avg_conceded": 0.6, "results": ["W","W","W","W","W"]},  # Inter — dominant
    489:  {"avg_scored": 1.4, "avg_conceded": 1.2, "results": ["D","W","L","D","W"]},  # AC Milan
    496:  {"avg_scored": 1.2, "avg_conceded": 1.0, "results": ["D","D","W","D","D"]},  # Juventus
    492:  {"avg_scored": 2.0, "avg_conceded": 0.8, "results": ["W","W","W","D","L"]},  # Napoli
    157:  {"avg_scored": 3.2, "avg_conceded": 1.0, "results": ["W","W","W","W","W"]},  # Bayern
    165:  {"avg_scored": 1.6, "avg_conceded": 1.8, "results": ["L","W","D","L","W"]},  # Dortmund
    168:  {"avg_scored": 2.0, "avg_conceded": 0.8, "results": ["W","W","W","D","W"]},  # Leverkusen
    173:  {"avg_scored": 1.4, "avg_conceded": 1.4, "results": ["D","W","L","W","D"]},  # RB Leipzig
    85:   {"avg_scored": 3.4, "avg_conceded": 0.4, "results": ["W","W","W","W","W"]},  # PSG
    91:   {"avg_scored": 1.8, "avg_conceded": 1.0, "results": ["W","D","W","W","D"]},  # Monaco
    80:   {"avg_scored": 0.8, "avg_conceded": 1.6, "results": ["L","D","L","D","W"]},  # Lyon — poor
    81:   {"avg_scored": 1.6, "avg_conceded": 1.2, "results": ["W","D","D","W","L"]},  # Marseille
    194:  {"avg_scored": 2.0, "avg_conceded": 1.0, "results": ["W","W","D","W","L"]},  # Ajax
    497:  {"avg_scored": 1.6, "avg_conceded": 0.8, "results": ["W","D","W","D","W"]},  # Roma
    6:    {"avg_scored": 2.2, "avg_conceded": 0.8, "results": ["W","W","D","W","W"]},  # Brazil
    26:   {"avg_scored": 2.6, "avg_conceded": 0.6, "results": ["W","W","W","W","D"]},  # Argentina
    2:    {"avg_scored": 1.8, "avg_conceded": 1.0, "results": ["W","D","W","L","W"]},  # France
    25:   {"avg_scored": 2.0, "avg_conceded": 0.8, "results": ["W","W","D","W","D"]},  # Germany
    9:    {"avg_scored": 2.4, "avg_conceded": 0.6, "results": ["W","W","W","D","W"]},  # Spain
    10:   {"avg_scored": 1.6, "avg_conceded": 1.0, "results": ["D","W","W","D","W"]},  # England
    27:   {"avg_scored": 2.0, "avg_conceded": 0.8, "results": ["W","W","D","W","W"]},  # Portugal
    1580: {"avg_scored": 1.0, "avg_conceded": 0.8, "results": ["W","D","D","W","L"]},  # Morocco
}

_NO_INJ = {"attack_factor": 1.0, "defense_factor": 1.0, "players": []}

DEMO_INJURIES = {
    50:  {"attack_factor": 0.91, "defense_factor": 1.06,
          "players": [{"name": "K. De Bruyne", "position": "Midfielder", "reason": "Injured"}]},
    33:  {"attack_factor": 0.86, "defense_factor": 1.08,
          "players": [{"name": "R. Højlund",   "position": "Attacker",   "reason": "Injured"},
                      {"name": "L. Shaw",       "position": "Defender",   "reason": "Injured"}]},
    536: {"attack_factor": 0.93, "defense_factor": 1.05,
          "players": [{"name": "Y. En-Nesyri", "position": "Attacker",   "reason": "Injured"}]},
    165: {"attack_factor": 0.93, "defense_factor": 1.04,
          "players": [{"name": "N. Füllkrug",  "position": "Attacker",   "reason": "Injured"}]},
    10:  {"attack_factor": 0.94, "defense_factor": 1.03,
          "players": [{"name": "B. Saka",      "position": "Attacker",   "reason": "Suspended"}]},
}


def demo_injury(team_id: int) -> dict:
    return DEMO_INJURIES.get(team_id, _NO_INJ)


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
    _odds_event("Liverpool",        "Manchester City",   2.20, 3.40, 3.10),
    _odds_event("Arsenal",          "Chelsea",           2.10, 3.30, 3.50),
    _odds_event("Manchester Utd",   "Tottenham",         2.70, 3.20, 2.60),
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
