#!/usr/bin/env python3
# travel_buddy.py
# AI Weekend Travel Buddy — Multi-Agent single-file solution (Explorer, Food, Budget)
# Works online (Overpass/OSRM) and offline (demo). Produces human + JSON outputs.
# Dependencies: requests (pip install requests)

import argparse, json, math, os, sys, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
try:
    import requests
except ImportError:
    print("Please `pip install requests`")
    sys.exit(1)

# ---------------------------
# Utilities
# ---------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

def haversine_km(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def clamp(v, lo, hi): return max(lo, min(hi, v))

def pretty_money(v, curr):
    if curr.upper() in ["INR", "₹", "RS", "RUPEE", "RUPEES"]:
        return f"₹{int(round(v)):,}"
    if curr.upper() in ["USD", "$"]:
        return f"${round(v,2):,.2f}"
    return f"{round(v,2):,.2f} {curr}"

def try_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

# ---------------------------
# Geocoding (simple, free, lightweight)
# ---------------------------

def geocode_nominatim(place: str) -> Optional[Tuple[float,float,str]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    try:
        r = requests.get(url, params=params, headers={"User-Agent":"ai-weekend-buddy/1.0"}, timeout=15)
        r.raise_for_status()
        arr = r.json()
        if arr:
            lat = float(arr[0]["lat"]); lon = float(arr[0]["lon"]); disp = arr[0].get("display_name", place)
            return (lat, lon, disp)
    except Exception:
        return None
    return None

# ---------------------------
# Overpass helpers (POIs)
# ---------------------------

def overpass_query(bbox: Tuple[float,float,float,float], filters: List[str], limit: int=40) -> List[Dict[str,Any]]:
    # bbox = (south, west, north, east)
    # filters: e.g. ['tourism=museum','historic=monument']
    # We fetch nodes & ways; dedupe by name/coords.
    f_or = "|".join(filters)
    query = f"""
    [out:json][timeout:25];
    (
      node[{f_or}]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      way[{f_or}]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out center {limit};
    """
    try:
        r = requests.post(OVERPASS_URL, data=query, timeout=40)
        r.raise_for_status()
        data = r.json()
        return data.get("elements", [])[:limit]
    except Exception:
        return []

def bbox_from_center(lat, lon, km=8):
    # Rough square bbox around center
    dlat = km/110.574
    dlon = km/(111.320*math.cos(math.radians(lat)))
    return (lat-dlat, lon-dlon, lat+dlat, lon+dlon)

def to_point(el):
    if "lat" in el and "lon" in el:
        return (el["lat"], el["lon"])
    if "center" in el and "lat" in el["center"] and "lon" in el["center"]:
        return (el["center"]["lat"], el["center"]["lon"])
    return None

# ---------------------------
# OSRM helpers (travel time)
# ---------------------------

def osrm_eta_minutes(a: Tuple[float,float], b: Tuple[float,float]) -> Optional[float]:
    try:
        url = f"{OSRM_URL}/{a[1]},{a[0]};{b[1]},{b[0]}"
        r = requests.get(url, params={"overview":"false"}, timeout=15)
        r.raise_for_status()
        routes = r.json().get("routes", [])
        if routes:
            sec = routes[0].get("duration", None)
            if sec is not None: return sec/60.0
    except Exception:
        pass
    # Fallback: driving ~ 35 km/h average in cities
    km = haversine_km(a,b); return (km/35.0)*60.0

# ---------------------------
# Data classes
# ---------------------------

@dataclass
class POI:
    name: str
    lat: float
    lon: float
    kind: str
    opening: Optional[str] = None
    fee: Optional[float] = None
    rating_hint: float = 0.0
    dwell_min: int = 75

@dataclass
class Eatery:
    name: str
    lat: float
    lon: float
    kind: str
    must_try: Optional[str] = None
    price_hint: float = 0.0

@dataclass
class SlotItem:
    when: str     # "morning"/"afternoon"/"evening"
    title: str
    lat: float
    lon: float
    note: str
    travel_min: float
    dwell_min: int
    est_cost: float

@dataclass
class DayPlan:
    day: int
    items: List[SlotItem]

@dataclass
class Itinerary:
    destination: str
    currency: str
    budget_total: float
    budget_breakdown: Dict[str, float]
    days: List[DayPlan]
    tips: List[str]

# ---------------------------
# Agents
# ---------------------------

class ExplorerAgent:
    ATTRACTION_FILTERS = [
        'tourism=museum', 'tourism=attraction', 'tourism=gallery',
        'historic=monument', 'historic=castle', 'historic=ruins',
        'leisure=park', 'amenity=place_of_worship', 'shop=market'
    ]

    def __init__(self, center: Tuple[float,float], bbox, prefs: List[str]):
        self.center = center
        self.bbox = bbox
        self.prefs = [p.strip().lower() for p in prefs if p.strip()]

    def fetch(self, online=True) -> List[POI]:
        if not online:
            # Minimal demo data
            base = [
                POI("City Museum", self.center[0]+0.01, self.center[1]+0.01, "museum", "10:00-18:00", 5.0, 0.8, 90),
                POI("Old Fort", self.center[0]+0.015, self.center[1]-0.008, "fort", "06:00-18:00", 3.0, 0.7, 75),
                POI("Central Park", self.center[0]-0.01, self.center[1]+0.005, "park", "Open 24h", 0.0, 0.6, 60),
                POI("Bazaar Market", self.center[0]-0.008, self.center[1]-0.01, "market", "10:00-22:00", 0.0, 0.75, 80),
                POI("Riverside Walk", self.center[0]+0.005, self.center[1], "walk", "Open 24h", 0.0, 0.5, 50),
            ]
            return base

        els = overpass_query(self.bbox, self.ATTRACTION_FILTERS, limit=60)
        pois: List[POI] = []
        seen = set()
        for el in els:
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name: continue
            pt = to_point(el); if pt is None: continue
            key = (round(pt[0],5), round(pt[1],5), name.lower())
            if key in seen: continue
            seen.add(key)
            kind = (tags.get("tourism") or tags.get("historic") or tags.get("leisure") or tags.get("shop") or "attraction")
            opening = tags.get("opening_hours")
            fee = None
            if "fee" in tags:
                fee = 0.0 if tags["fee"].lower() in ["no","0","free"] else 5.0
            # Preference scoring hint
            base = 0.6
            name_l = name.lower()
            for pref in self.prefs:
                if pref in name_l or pref in kind: base += 0.15
            base = clamp(base, 0.4, 0.95)
            dwell = 60 if kind in ["park", "market"] else 75
            pois.append(POI(name, pt[0], pt[1], kind, opening, fee, base, dwell))
        # Rank by rating_hint + proximity
        for p in pois:
            dist_km = haversine_km(self.center, (p.lat,p.lon))
            p.rating_hint = p.rating_hint + clamp(1.2 - 0.08*dist_km, 0, 1)
        pois.sort(key=lambda x: x.rating_hint, reverse=True)
        return pois[:18]

    def sequence_two_days(self, pois: List[POI]) -> Dict[int, List[POI]]:
        # Greedy: alternates top picks across two days, keeping mix
        day1, day2 = [], []
        for i, poi in enumerate(pois):
            (day1 if i%2==0 else day2).append(poi)
        # Keep at most ~6 per day
        return {1: day1[:6], 2: day2[:6]}

class FoodAgent:
    FOOD_FILTERS = [
        'amenity=restaurant', 'amenity=cafe', 'amenity=fast_food', 'amenity=food_court', 'amenity=street_food'
    ]
    MUST_TRY_HINTS = ["chole bhature","chaat","kebabs","parantha","biryani","jalebi","samosa","vada pav","pani puri","dosa"]

    def __init__(self, bbox):
        self.bbox = bbox

    def fetch_near(self, lat, lon, online=True) -> List[Eatery]:
        if not online:
            return [
                Eatery("Corner Café", lat+0.001, lon+0.001, "cafe", "Fresh pastries", 6.0),
                Eatery("Spice Street", lat+0.0005, lon-0.0012, "street_food", "Local chaat", 4.0),
                Eatery("Heritage Bites", lat-0.001, lon+0.0007, "restaurant", "Regional thali", 8.0),
            ]
        els = overpass_query(self.bbox, self.FOOD_FILTERS, limit=80)
        eats = []
        for el in els:
            tags = el.get("tags", {})
            name = tags.get("name"); 
            if not name: continue
            pt = to_point(el); 
            if pt is None: continue
            km = haversine_km((lat,lon), pt)
            if km > 1.0:  # keep within 1 km
                continue
            kind = (tags.get("amenity") or "food")
            # naive price hint
            price = 5.0 if kind in ["street_food","fast_food","cafe"] else 10.0
            # cheeky must-try from name
            mt = None
            lower = name.lower()
            for hint in self.MUST_TRY_HINTS:
                if hint in lower:
                    mt = hint; break
            eats.append(Eatery(name, pt[0], pt[1], kind, mt, price))
        # sort by proximity then kind preference
        eats.sort(key=lambda e: (haversine_km((lat,lon),(e.lat,e.lon)), 0 if e.kind in ["street_food","restaurant"] else 1))
        return eats[:5]

class BudgetAgent:
    def __init__(self, total: float, currency: str):
        self.total = total
        self.currency = currency

    def split(self) -> Dict[str,float]:
        # Default splits; you can tune per destination later
        return {
            "stay": 0.40*self.total,
            "food": 0.30*self.total,
            "activities": 0.20*self.total,
            "local_travel": 0.10*self.total,
        }

    def price_activity(self, poi: POI) -> float:
        if poi.fee is not None:
            return poi.fee
        # heuristic by kind
        table = {"museum": 8, "gallery": 5, "castle": 10, "ruins": 4, "park": 0, "market": 0, "attraction": 5, "fort": 6}
        return float(table.get(poi.kind, 4))

    def price_food_meal(self, eat: Eatery) -> float:
        base = {"street_food": 3, "fast_food": 5, "cafe": 6, "restaurant": 12, "food_court": 8}
        return float(base.get(eat.kind, 8))

# ---------------------------
# Orchestrator
# ---------------------------

class Orchestrator:
    def __init__(self, destination: str, budget: float, currency: str, prefs: List[str], start_date: Optional[str], offline_demo: bool):
        self.destination = destination
        self.budget = budget
        self.currency = currency
        self.prefs = prefs
        self.start_date = start_date
        self.offline_demo = offline_demo
        geoc = (28.6139, 77.2090, destination) if offline_demo else geocode_nominatim(destination)  # Delhi default if demo
        if not geoc:
            raise ValueError("Could not geocode destination.")
        self.center = (geoc[0], geoc[1])
        self.display_name = geoc[2]
        self.bbox = bbox_from_center(*self.center, km=8)

        self.explorer = ExplorerAgent(self.center, self.bbox, prefs)
        self.food = FoodAgent(self.bbox)
        self.budget_agent = BudgetAgent(budget, currency)

    def build(self) -> Itinerary:
        online = not self.offline_demo
        breakdown = self.budget_agent.split()

        pois = self.explorer.fetch(online=online)
        if not pois:
            # Safety fallback
            pois = self.explorer.fetch(online=False)

        days_map = self.explorer.sequence_two_days(pois)

        day_plans: List[DayPlan] = []
        # Simple slot plan: 3 slots per day
        slots = ["morning","afternoon","evening"]
        local_travel_spend = 0.0
        food_spend = 0.0
        act_spend = 0.0

        for d in [1,2]:
            items: List[SlotItem] = []
            day_pois = days_map.get(d, [])[:3]  # at most 3 main stops per day
            prev_point = self.center
            for idx, (slot, poi) in enumerate(zip(slots, day_pois)):
                eta = osrm_eta_minutes(prev_point, (poi.lat, poi.lon)) if online else (haversine_km(prev_point,(poi.lat,poi.lon))/30.0)*60.0
                local_travel_spend += clamp(eta/15.0, 0, 3)  # rough cost proxy
                # Food near this POI
                eats = self.food.fetch_near(poi.lat, poi.lon, online=online)
                eat_note = ""
                if eats:
                    pick = eats[0]
                    cost = self.budget_agent.price_food_meal(pick)
                    food_spend += cost
                    must = f" (must-try: {pick.must_try})" if pick.must_try else ""
                    eat_note = f" Nearby food: {pick.name}{must} ~ {pretty_money(cost, self.currency)}"
                act_cost = self.budget_agent.price_activity(poi)
                act_spend += act_cost
                note = f"{poi.kind.title()}"
                if poi.opening: note += f" · Hours: {poi.opening}"
                if act_cost>0: note += f" · Entry ~ {pretty_money(act_cost, self.currency)}"
                note += eat_note
                items.append(SlotItem(slot, poi.name, poi.lat, poi.lon, note, eta, poi.dwell_min, act_cost))
                prev_point = (poi.lat, poi.lon)

            # Add evening “free time / market” if we had fewer than 3 POIs
            while len(items) < 3:
                eta = osrm_eta_minutes(prev_point, self.center) if online else 15.0
                local_travel_spend += clamp(eta/20.0, 0, 2)
                items.append(SlotItem(slots[len(items)], "Leisure time / Local Market",
                                      prev_point[0], prev_point[1],
                                      "Explore a nearby market or promenade; shop for souvenirs.",
                                      eta, 60, 0.0))
            day_plans.append(DayPlan(d, items))

        # Budget enforcement (coarse): if we overshoot food or activities, trim by adding more parks/markets and cheaper eats
        # (For hackathon scope, we only check and warn.)
        tips = []
        if food_spend > breakdown["food"] * 1.15:
            tips.append("Food plan is trending high; consider 1–2 street-food meals to rebalance.")
        if act_spend > breakdown["activities"] * 1.15:
            tips.append("Activities may exceed allocation; swap one paid site for a free park/market.")
        if (local_travel_spend*2) > breakdown["local_travel"] * 1.25:  # *2 to roughly convert proxy units to currency
            tips.append("Local travel costs are high; cluster attractions tighter or use metro/walking.")

        return Itinerary(self.display_name, self.currency, self.budget,
                         {"stay": round(breakdown["stay"],2),
                          "food": round(breakdown["food"],2),
                          "activities": round(breakdown["activities"],2),
                          "local_travel": round(breakdown["local_travel"],2)},
                         day_plans, tips)

    def render_human(self, it: Itinerary) -> str:
        lines = []
        lines.append(f"🎒 AI Weekend Travel Buddy — {it.destination}")
        lines.append(f"Budget: {pretty_money(it.budget_total, it.currency)}")
        bb = it.budget_breakdown
        lines.append("Budget split → " +
                     f"Stay {pretty_money(bb['stay'],it.currency)}, Food {pretty_money(bb['food'],it.currency)}, "
                     f"Activities {pretty_money(bb['activities'],it.currency)}, Local Travel {pretty_money(bb['local_travel'],it.currency)}")
        lines.append("")
        for day in it.days:
            lines.append(f"Day {day.day}")
            for s in day.items:
                lines.append(f"  • {s.when.title()}: {s.title}")
                lines.append(f"    ⏱ Travel ~ {int(round(s.travel_min))} min · Dwell ~ {s.dwell_min} min")
                if s.est_cost>0:
                    lines.append(f"    💵 {pretty_money(s.est_cost, it.currency)}")
                lines.append(f"    📝 {s.note}")
            lines.append("")
        if it.tips:
            lines.append("💡 Budget Tips:")
            for t in it.tips: lines.append(f"  – {t}")
        return "\n".join(lines)

    def render_json(self, it: Itinerary) -> Dict[str,Any]:
        return {
            "destination": it.destination,
            "currency": it.currency,
            "budget_total": it.budget_total,
            "budget_breakdown": it.budget_breakdown,
            "days": [
                {
                    "day": d.day,
                    "items": [asdict(i) for i in d.items]
                } for d in it.days
            ],
            "tips": it.tips
        }

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="AI Weekend Travel Buddy (multi-agent, single file)")
    ap.add_argument("--destination", required=True, help="City or place name")
    ap.add_argument("--budget", type=float, required=True, help="Total budget for 2 days")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--prefs", default="", help="Comma-separated interests (e.g., 'history, street food, markets')")
    ap.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) – optional (display only)")
    ap.add_argument("--offline_demo", action="store_true", help="Use offline demo data (no internet calls)")
    args = ap.parse_args()

    prefs = [p.strip() for p in args.prefs.split(",")] if args.prefs else []
    orch = Orchestrator(args.destination, args.budget, args.currency, prefs, args.start, args.offline_demo)
    it = orch.build()
    human = orch.render_human(it)
    print(human)
    with open("itinerary.json","w",encoding="utf-8") as f:
        json.dump(orch.render_json(it), f, ensure_ascii=False, indent=2)
    print("\nSaved JSON → itinerary.json")

if __name__ == "__main__":
    main()
