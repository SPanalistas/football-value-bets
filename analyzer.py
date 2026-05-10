"""
Football Value Bet Analyzer
Uses Poisson model + real odds to detect value bets
"""

import os
import sys
import math
import requests
import pandas as pd
from datetime import datetime, date
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIG ───────────────────────────────────────────────────────────────────

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_FOOTBALL_DATA_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_ODDS_API_KEY")

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Competitions to analyze (football-data.org competition codes)
COMPETITIONS = {
    "PL":  "Premier League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
    "EL":  "Europa League",
}

# The Odds API sport keys
SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
]

VALUE_THRESHOLD = 0.05   # Minimum edge to flag as value bet (5%)
MIN_MATCHES_FOR_MODEL = 5  # Minimum historical matches needed per team


# ─── FOOTBALL-DATA.ORG CLIENT ─────────────────────────────────────────────────

class FootballDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{FOOTBALL_DATA_BASE}/{endpoint}"
        response = self.session.get(url, params=params, timeout=15)
        if response.status_code == 429:
            print("⚠️  football-data.org rate limit reached. Using cached/demo data.")
            return {}
        response.raise_for_status()
        return response.json()

    def get_today_matches(self) -> list[dict]:
        """Fetch all matches scheduled for today across tracked competitions."""
        today = date.today().isoformat()
        matches = []
        for code, name in COMPETITIONS.items():
            try:
                data = self._get(f"competitions/{code}/matches", {
                    "dateFrom": today,
                    "dateTo": today,
                    "status": "SCHEDULED,TIMED,IN_PLAY"
                })
                for m in data.get("matches", []):
                    m["_competition"] = name
                    m["_comp_code"] = code
                    matches.append(m)
            except requests.HTTPError as e:
                print(f"  ⚠️  Could not fetch {name}: {e}")
        return matches

    def get_team_last_matches(self, team_id: int, limit: int = 20) -> list[dict]:
        """Fetch last N finished matches for a team."""
        try:
            data = self._get(f"teams/{team_id}/matches", {
                "status": "FINISHED",
                "limit": limit,
            })
            return data.get("matches", [])
        except requests.HTTPError:
            return []


# ─── POISSON MODEL ────────────────────────────────────────────────────────────

class PoissonModel:
    """
    Estimate home/away attack & defense strength from historical results,
    then compute 1X2 probabilities via Poisson distribution.
    """

    def __init__(self):
        self._avg_home_goals = 1.5
        self._avg_away_goals = 1.2

    def _team_stats(self, matches: list[dict], team_id: int) -> dict:
        """Goals scored/conceded split by home/away from match list."""
        home_scored, home_conceded, home_n = 0, 0, 0
        away_scored, away_conceded, away_n = 0, 0, 0

        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            home_goals = score.get("home")
            away_goals = score.get("away")
            if home_goals is None or away_goals is None:
                continue
            home_id = m.get("homeTeam", {}).get("id")
            away_id = m.get("awayTeam", {}).get("id")
            if home_id == team_id:
                home_scored += home_goals
                home_conceded += away_goals
                home_n += 1
            elif away_id == team_id:
                away_scored += away_goals
                away_conceded += home_goals
                away_n += 1

        return {
            "home_scored": home_scored,
            "home_conceded": home_conceded,
            "home_n": home_n,
            "away_scored": away_scored,
            "away_conceded": away_conceded,
            "away_n": away_n,
        }

    def _attack_defense(self, stats: dict, as_home: bool) -> tuple[float, float]:
        """Return (attack_strength, defense_strength) for a team's role."""
        if as_home:
            n = stats["home_n"]
            scored = stats["home_scored"] / n if n else self._avg_home_goals
            conceded = stats["home_conceded"] / n if n else self._avg_away_goals
            attack = scored / self._avg_home_goals
            defense = conceded / self._avg_away_goals
        else:
            n = stats["away_n"]
            scored = stats["away_scored"] / n if n else self._avg_away_goals
            conceded = stats["away_conceded"] / n if n else self._avg_home_goals
            attack = scored / self._avg_away_goals
            defense = conceded / self._avg_home_goals
        return max(attack, 0.1), max(defense, 0.1)

    def expected_goals(
        self,
        home_matches: list[dict],
        away_matches: list[dict],
        home_id: int,
        away_id: int,
    ) -> tuple[float, float]:
        """Estimate λ_home and λ_away (expected goals) for the fixture."""
        h_stats = self._team_stats(home_matches, home_id)
        a_stats = self._team_stats(away_matches, away_id)

        h_attack, h_defense = self._attack_defense(h_stats, as_home=True)
        a_attack, a_defense = self._attack_defense(a_stats, as_home=False)

        lambda_home = h_attack * a_defense * self._avg_home_goals
        lambda_away = a_attack * h_defense * self._avg_away_goals

        return round(lambda_home, 3), round(lambda_away, 3)

    def goal_matrix(self, lam_h: float, lam_a: float, max_goals: int = 8) -> pd.DataFrame:
        """Build a (max_goals+1)×(max_goals+1) probability matrix."""
        rows = []
        for h in range(max_goals + 1):
            row = []
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
                row.append(p)
            rows.append(row)
        return pd.DataFrame(rows)

    def match_probabilities(self, lam_h: float, lam_a: float) -> dict[str, float]:
        """Return P(home win), P(draw), P(away win) from Poisson model."""
        matrix = self.goal_matrix(lam_h, lam_a)
        arr = matrix.values
        n = arr.shape[0]
        p_home_win = sum(arr[h, a] for h in range(n) for a in range(n) if h > a)
        p_draw     = sum(arr[h, a] for h in range(n) for a in range(n) if h == a)
        p_away_win = sum(arr[h, a] for h in range(n) for a in range(n) if h < a)

        total = p_home_win + p_draw + p_away_win
        return {
            "home_win": p_home_win / total,
            "draw":     p_draw / total,
            "away_win": p_away_win / total,
        }


# ─── THE ODDS API CLIENT ──────────────────────────────────────────────────────

class OddsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_odds(self, sport_key: str) -> list[dict]:
        """Fetch H2H (1X2) odds for a sport/competition."""
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            remaining = r.headers.get("x-requests-remaining", "?")
            print(f"  ✅  {sport_key}: {len(r.json())} events | Requests left: {remaining}")
            return r.json()
        except requests.HTTPError as e:
            print(f"  ⚠️  Odds unavailable for {sport_key}: {e}")
            return []

    def best_odds(self, event: dict) -> dict[str, float]:
        """Extract best available decimal odds across bookmakers for 1X2."""
        best = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
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
    """Convert decimal odds to implied probability."""
    return 1.0 / decimal_odds if decimal_odds > 1 else 0.0


def edge(model_prob: float, decimal_odds: float) -> float:
    """Expected value edge = model_prob × odds − 1."""
    return model_prob * decimal_odds - 1.0


def detect_value_bets(model_probs: dict, best_odds: dict, threshold: float = VALUE_THRESHOLD) -> list[dict]:
    """Return list of outcomes where model edge > threshold."""
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
        ev = edge(prob, odds)
        if ev >= threshold:
            bets.append({
                "outcome": label,
                "model_prob": prob,
                "implied_prob": implied_probability(odds),
                "best_odds": odds,
                "edge": ev,
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
        if a == b:
            return True
        if a in b or b in a:
            return True
        ta, tb = set(a.split()), set(b.split())
        shorter = ta if len(ta) <= len(tb) else tb
        if shorter and len(ta & tb) >= max(1, len(shorter) - 1):
            return True
        return len(a) >= 4 and len(b) >= 4 and a[:5] == b[:5]

    return similar(fd_home, odds_home) and similar(fd_away, odds_away)


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def print_header():
    print("\n" + "═" * 70)
    print("  ⚽  FOOTBALL VALUE BET ANALYZER")
    print(f"  📅  {datetime.now().strftime('%A, %d %B %Y  %H:%M')}")
    print("═" * 70)


def print_match_result(match_info: dict):
    home = match_info["home_team"]
    away = match_info["away_team"]
    comp = match_info["competition"]
    kickoff = match_info.get("kickoff", "")
    lam_h = match_info["lambda_home"]
    lam_a = match_info["lambda_away"]
    probs = match_info["model_probs"]
    bets = match_info["value_bets"]
    has_odds = match_info.get("has_odds", False)

    print(f"\n  🏆  {comp}")
    print(f"  {home}  vs  {away}", end="")
    if kickoff:
        print(f"  |  {kickoff}", end="")
    print()
    print(f"  Expected goals → {home}: {lam_h}  |  {away}: {lam_a}")
    print(f"  Model probs    → Home: {probs['home_win']:.1%}  "
          f"Draw: {probs['draw']:.1%}  Away: {probs['away_win']:.1%}")

    if not has_odds:
        print("  ⚠️  No odds data available for this match")
        return

    if bets:
        print(f"\n  🎯  VALUE BETS DETECTED:")
        for b in bets:
            bar = "▓" * int(b["edge"] * 100)
            print(f"     → {b['outcome']:10s}  "
                  f"Odds: {b['best_odds']:.2f}  "
                  f"Model: {b['model_prob']:.1%}  "
                  f"Implied: {b['implied_prob']:.1%}  "
                  f"Edge: +{b['edge']:.1%}  {bar}")
    else:
        print("  ✗  No value bets found (market appears fairly priced)")

    print("  " + "─" * 66)


def print_summary(results: list[dict]):
    value_matches = [r for r in results if r["value_bets"]]
    total_bets = sum(len(r["value_bets"]) for r in results)

    print("\n" + "═" * 70)
    print(f"  📊  SUMMARY")
    print(f"  Matches analyzed : {len(results)}")
    print(f"  Matches with edge: {len(value_matches)}")
    print(f"  Total value bets : {total_bets}")

    if value_matches:
        all_bets = []
        for r in value_matches:
            for b in r["value_bets"]:
                all_bets.append({
                    "Match": f"{r['home_team']} vs {r['away_team']}",
                    "Outcome": b["outcome"],
                    "Odds": b["best_odds"],
                    "Model %": f"{b['model_prob']:.1%}",
                    "Market %": f"{b['implied_prob']:.1%}",
                    "Edge": f"+{b['edge']:.1%}",
                })
        df = pd.DataFrame(all_bets).sort_values("Edge", ascending=False)
        print("\n  🏆  TOP VALUE BETS RANKING:")
        print(df.to_string(index=False))
    print("═" * 70 + "\n")


# ─── DEMO / FALLBACK DATA ─────────────────────────────────────────────────────

DEMO_MATCHES = [
    {
        "id": 1,
        "homeTeam": {"id": 64, "name": "Liverpool"},
        "awayTeam": {"id": 65, "name": "Manchester City"},
        "_competition": "Premier League",
        "_comp_code": "PL",
        "utcDate": f"{date.today().isoformat()}T15:00:00Z",
        "_demo": True,
    },
    {
        "id": 2,
        "homeTeam": {"id": 86, "name": "Real Madrid"},
        "awayTeam": {"id": 81, "name": "Barcelona"},
        "_competition": "La Liga",
        "_comp_code": "PD",
        "utcDate": f"{date.today().isoformat()}T20:00:00Z",
        "_demo": True,
    },
]

def _demo_history(team_id: int, home_results: list, away_results: list) -> list:
    """Generate demo match history for a team."""
    records = []
    for h, a in home_results:
        records.append({"score": {"fullTime": {"home": h, "away": a}},
                        "homeTeam": {"id": team_id}, "awayTeam": {"id": 999}})
    for h, a in away_results:
        records.append({"score": {"fullTime": {"home": h, "away": a}},
                        "homeTeam": {"id": 999}, "awayTeam": {"id": team_id}})
    return records

# Liverpool (64): strong home, decent away
_LIV = _demo_history(64,
    home_results=[(3,1),(2,0),(4,0),(2,1),(3,2)],
    away_results=[(1,1),(2,1),(0,1),(2,2),(1,0)])

# Man City (65): prolific scorer
_MCI = _demo_history(65,
    home_results=[(4,0),(3,1),(2,0),(3,2),(4,1)],
    away_results=[(2,1),(1,1),(3,1),(2,2),(1,0)])

# Real Madrid (86): strong home
_RMA = _demo_history(86,
    home_results=[(3,0),(2,1),(4,1),(1,0),(2,0)],
    away_results=[(1,1),(2,0),(0,2),(1,1),(2,1)])

# Barcelona (81): attack-minded
_BAR = _demo_history(81,
    home_results=[(4,1),(3,0),(2,1),(3,2),(2,0)],
    away_results=[(2,1),(1,2),(1,1),(0,1),(2,2)])

DEMO_HISTORY = {64: _LIV, 65: _MCI, 86: _RMA, 81: _BAR}

DEMO_ODDS = [
    {
        "home_team": "Liverpool",
        "away_team": "Manchester City",
        "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
            {"name": "Liverpool", "price": 2.40},
            {"name": "Draw", "price": 3.30},
            {"name": "Manchester City", "price": 2.90},
        ]}]}]
    },
    {
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
            {"name": "Real Madrid", "price": 2.10},
            {"name": "Draw", "price": 3.50},
            {"name": "Barcelona", "price": 3.20},
        ]}]}]
    },
]


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run():
    print_header()

    demo_mode = (
        FOOTBALL_DATA_API_KEY == "YOUR_FOOTBALL_DATA_KEY"
        or ODDS_API_KEY == "YOUR_ODDS_API_KEY"
    )

    if demo_mode:
        print("\n  ℹ️  DEMO MODE — set env vars FOOTBALL_DATA_API_KEY & ODDS_API_KEY")
        print("        to use live data.\n")

    # 1. Fetch today's matches
    print("📡  Fetching today's matches …")
    if demo_mode:
        today_matches = DEMO_MATCHES
        print(f"  → {len(today_matches)} demo matches loaded")
    else:
        fd_client = FootballDataClient(FOOTBALL_DATA_API_KEY)
        today_matches = fd_client.get_today_matches()
        print(f"  → {len(today_matches)} matches found")

    if not today_matches:
        print("\n  No matches scheduled for today. Try again tomorrow! ⚽")
        return

    # 2. Fetch all odds (batch per sport key)
    print("\n📡  Fetching odds from The Odds API …")
    all_odds_events: list[dict] = []
    if demo_mode:
        all_odds_events = DEMO_ODDS
    else:
        odds_client = OddsClient(ODDS_API_KEY)
        for sport_key in SPORT_KEYS:
            all_odds_events.extend(odds_client.get_odds(sport_key))

    print(f"  → {len(all_odds_events)} odds events loaded")

    # 3. Build OddsClient outside demo for best_odds method
    if not demo_mode:
        odds_client_inst = OddsClient(ODDS_API_KEY)
    else:
        odds_client_inst = None

    model = PoissonModel()

    print("\n🔢  Running Poisson model & value bet detection …\n")

    results = []

    for match in today_matches:
        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        home_id   = match["homeTeam"]["id"]
        away_id   = match["awayTeam"]["id"]
        comp      = match["_competition"]
        kickoff   = match.get("utcDate", "")[:16].replace("T", " ") + " UTC"

        # 3a. Get historical data
        if demo_mode:
            home_hist = DEMO_HISTORY.get(home_id, [])
            away_hist = DEMO_HISTORY.get(away_id, [])
        else:
            home_hist = fd_client.get_team_last_matches(home_id)
            away_hist = fd_client.get_team_last_matches(away_id)

        if len(home_hist) < MIN_MATCHES_FOR_MODEL or len(away_hist) < MIN_MATCHES_FOR_MODEL:
            print(f"  ⚠️  Skipping {home_team} vs {away_team} — insufficient history")
            continue

        # 3b. Poisson model
        lam_h, lam_a = model.expected_goals(home_hist, away_hist, home_id, away_id)
        probs = model.match_probabilities(lam_h, lam_a)

        # 3c. Match to odds event
        matched_event = None
        for ev in all_odds_events:
            if teams_match(home_team, away_team, ev.get("home_team", ""), ev.get("away_team", "")):
                matched_event = ev
                break

        if matched_event:
            if demo_mode:
                # Manual best odds for demo
                best = {"home": 0.0, "draw": 0.0, "away": 0.0}
                for bm in matched_event.get("bookmakers", []):
                    for market in bm.get("markets", []):
                        for outcome in market.get("outcomes", []):
                            n, p = outcome.get("name",""), outcome.get("price", 0)
                            if n == home_team:    best["home"] = max(best["home"], p)
                            elif n == away_team:  best["away"] = max(best["away"], p)
                            elif n == "Draw":     best["draw"] = max(best["draw"], p)
            else:
                best = odds_client_inst.best_odds(matched_event)
            bets = detect_value_bets(probs, best)
            has_odds = True
        else:
            best = {}
            bets = []
            has_odds = False

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "competition": comp,
            "kickoff": kickoff,
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "model_probs": probs,
            "best_odds": best,
            "value_bets": bets,
            "has_odds": has_odds,
        }
        results.append(result)
        print_match_result(result)

    print_summary(results)


if __name__ == "__main__":
    run()
