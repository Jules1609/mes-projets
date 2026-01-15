# backend/solver.py
from ortools.sat.python import cp_model
import datetime
from typing import Dict, Any, List

# tu gardes ton cost()
from data import cost

class Medecin:
    def __init__(self, name, max_jours, jours_interdits=None, salles_autorisees=None, vacances=None):
        self.name = name
        self.max_jours = max_jours
        self.jours_interdits = jours_interdits or []      # list[(jour, plage)] ex ("Lundi","M")
        self.salles_autorisees = salles_autorisees or []  # list[str]
        self.vacances = vacances or []                    # list[date]

jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
plages = ["M", "A"]  # Matin / Après-midi

def _normalize_slot_token(token: str) -> str:
    """Convert UI tokens like 'Lundi matin'/'Lundi après-midi' or 'Lundi AM/PM' to solver tokens 'Lundi M'/'Lundi A'."""
    t = (token or "").strip()
    if not t:
        return ""

    # Already in solver format?
    # Examples: 'Lundi M', 'Mardi A'
    parts = t.split()
    if len(parts) == 2 and parts[0] in jours and parts[1] in plages:
        return t

    # Common UI formats
    lower = t.lower()
    # Replace various separators
    lower = lower.replace("-", " ").replace("_", " ")

    # Day detection
    day = None
    for j in jours:
        if lower.startswith(j.lower()):
            day = j
            break

    if not day:
        return ""

    # Slot detection
    if "matin" in lower or lower.endswith(" am") or lower.endswith(" a.m") or " am" in lower:
        return f"{day} M"
    if "après" in lower or "apres" in lower or lower.endswith(" pm") or " pm" in lower:
        return f"{day} A"

    # Fallback: try last token
    last = parts[-1].upper()
    if last in ("AM", "M"):
        return f"{day} M"
    if last in ("PM", "A"):
        return f"{day} A"

    return ""


def _normalize_disponibilites_salles(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """Accept both:
    - disponibilites_salles: {room: ['Lundi M', ...]}
    - rooms_availability: {room: ['Lundi matin', ...]}
    Returns: {room: ['Lundi M', ...]}.
    """
    raw = payload.get("disponibilites_salles")
    if raw is None:
        raw = payload.get("rooms_availability")

    if not isinstance(raw, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for room, tokens in raw.items():
        if tokens is None:
            tokens = []
        if not isinstance(tokens, list):
            # allow a set/tuple or a single string
            if isinstance(tokens, (set, tuple)):
                tokens = list(tokens)
            elif isinstance(tokens, str):
                tokens = [tokens]
            else:
                tokens = []

        norm = []
        for tok in tokens:
            nt = _normalize_slot_token(str(tok))
            if nt:
                norm.append(nt)
        out[str(room)] = norm

    return out


def _normalize_doctors(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept both:
    - doctors: [ {name, max_jours, ...} ]
    - doctors: { 'Dr X': { ... } }
    Returns a list of doctor dicts.
    """
    docs = payload.get("doctors")
    if docs is None:
        return []

    if isinstance(docs, list):
        return docs

    if isinstance(docs, dict):
        out = []
        for name, d in docs.items():
            if isinstance(d, dict):
                dd = dict(d)
                dd.setdefault("name", name)
                out.append(dd)
        return out

    return []

def _parse_date(s: str) -> datetime.date:
    # accepte "2026-08-10"
    return datetime.date.fromisoformat(s)

def _date_to_wj(d: datetime.date, date_debut: datetime.date) -> tuple[int,int] | None:
    delta = (d - date_debut).days
    if delta < 0:
        return None
    w = delta // 7
    j = delta % 7
    # on ignore Dimanche (j==6) car tes jours = Lun..Sam (6 jours)
    if j >= len(jours):
        return None
    return (w, j)

def solve_planning(payload: Dict[str, Any]) -> Dict[str, Any]:
    year = int(payload.get("year", 2026))
    date_debut = datetime.date(year, 1, 1)

    # --- Parse médecins ---
    medecins: List[Medecin] = []
    doctors_payload = _normalize_doctors(payload)
    if not doctors_payload:
        return {"status": "error", "message": "Payload invalide : 'doctors' manquant ou vide."}

    for doc in doctors_payload:
        vacances = [_parse_date(d) for d in doc.get("vacances", [])]
        medecins.append(
            Medecin(
                name=doc["name"],
                max_jours=int(doc["max_jours"]),
                jours_interdits=[tuple(x) for x in doc.get("jours_interdits", [])],  # [["Lundi","M"], ...]
                salles_autorisees=list(doc.get("salles_autorisees", [])),
                vacances=vacances,
            )
        )

    # --- Parse salles + disponibilités ---
    dispo_norm = _normalize_disponibilites_salles(payload)

    salles = payload.get("salles")
    if salles is None:
        # If not provided, infer from dispo keys
        salles = list(dispo_norm.keys())

    if not isinstance(salles, list) or not salles:
        return {"status": "error", "message": "Payload invalide : 'salles' manquant/vide (ou aucune disponibilité fournie)."}

    salles = [str(s) for s in salles]

    disponibilites_salles: Dict[str, set] = {}
    for s in salles:
        disponibilites_salles[s] = set(dispo_norm.get(s, []))

    M = len(medecins)
    J = len(jours)
    P = len(plages)
    S = len(salles)

    # --- Rolling horizon ---
    # 52 semaines environ (attention: 2026-01-01 n'est pas forcément un lundi, mais on reste cohérent avec ton mapping)
    total_days = (datetime.date(year, 12, 31) - date_debut).days + 1
    total_weeks = (total_days + 6) // 7

    BLOCK_WEEKS = int(payload.get("block_weeks", 4))  # 4 semaines par défaut
    assignments_all: List[Dict[str, Any]] = []

    for block_start in range(0, total_weeks, BLOCK_WEEKS):
        W = min(BLOCK_WEEKS, total_weeks - block_start)

        model = cp_model.CpModel()

        # Variables x[m,w,j,p,s]
        x = {}
        for m in range(M):
            for w in range(W):
                for j in range(J):
                    for p in range(P):
                        for s_idx in range(S):
                            x[m, w, j, p, s_idx] = model.NewBoolVar(f"x_{m}_{w}_{j}_{p}_{s_idx}")

        # 1) 1 médecin par salle par plage si salle ouverte
        for w in range(W):
            for j in range(J):
                for p in range(P):
                    for s_idx in range(S):
                        jour_plage = f"{jours[j]} {plages[p]}"
                        salle_nom = salles[s_idx]
                        if jour_plage in disponibilites_salles.get(salle_nom, set()):
                            model.Add(sum(x[m, w, j, p, s_idx] for m in range(M)) == 1)
                        else:
                            model.Add(sum(x[m, w, j, p, s_idx] for m in range(M)) == 0)

        # 2) Un médecin ne peut pas être dans 2 salles au même slot (ton commentaire était inversé)
        for m in range(M):
            for w in range(W):
                for j in range(J):
                    for p in range(P):
                        model.Add(sum(x[m, w, j, p, s_idx] for s_idx in range(S)) <= 1)

        # 3) Jours/plages interdits
        for m in range(M):
            for (jour_interdit, plage_interdite) in medecins[m].jours_interdits:
                if jour_interdit in jours and plage_interdite in plages:
                    j_idx = jours.index(jour_interdit)
                    p_idx = plages.index(plage_interdite)
                    for w in range(W):
                        for s_idx in range(S):
                            model.Add(x[m, w, j_idx, p_idx, s_idx] == 0)

        # 4) Vacances (sur ce bloc uniquement)
        for m in range(M):
            for d in medecins[m].vacances:
                mapped = _date_to_wj(d, date_debut)
                if not mapped:
                    continue
                w_abs, j_idx = mapped
                # si la date tombe dans ce bloc
                if block_start <= w_abs < block_start + W:
                    w = w_abs - block_start
                    for p in range(P):
                        for s_idx in range(S):
                            model.Add(x[m, w, j_idx, p, s_idx] == 0)

        # 5) Max jours/semaine => tu l'as modélisé en demi-journées : <= 2*max_jours
        for m in range(M):
            for w in range(W):
                model.Add(
                    sum(x[m, w, j, p, s_idx] for j in range(J) for p in range(P) for s_idx in range(S))
                    <= 2 * medecins[m].max_jours
                )

        # 6) Restriction par salle + ouverture déjà gérée
        for m in range(M):
            allowed = set(medecins[m].salles_autorisees)
            for w in range(W):
                for j in range(J):
                    for p in range(P):
                        jour_plage = f"{jours[j]} {plages[p]}"
                        for s_idx in range(S):
                            salle_nom = salles[s_idx]
                            if (salle_nom not in allowed) or (jour_plage not in disponibilites_salles.get(salle_nom, set())):
                                model.Add(x[m, w, j, p, s_idx] == 0)

        # Objectif coût + équité (comme toi)
        cout_total = []
        for m in range(M):
            cout_total_m = sum(
                cost(medecins[m].name, jours[j], plages[p], salles[s_idx]) * x[m, w, j, p, s_idx]
                for w in range(W)
                for j in range(J)
                for p in range(P)
                for s_idx in range(S)
            )
            cout_total.append(cout_total_m)

        alpha = int(payload.get("alpha", 100))
        beta = int(payload.get("beta", 1))

        abs_diff_terms = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                diff = model.NewIntVar(-10_000_000, 10_000_000, f"diff_costtotal_{m1}_{m2}")
                absdiff = model.NewIntVar(0, 10_000_000, f"absdiff_costtotal_{m1}_{m2}")
                model.Add(diff == cout_total[m1] - cout_total[m2])
                model.AddAbsEquality(absdiff, diff)
                abs_diff_terms.append(absdiff)

        model.Minimize(alpha * sum(cout_total) + beta * sum(abs_diff_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(payload.get("max_time", 5))

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"status": "infeasible", "block_start_week": block_start, "message": "Pas de solution sur un bloc."}

        # Export JSON assignments de ce bloc
        for w in range(W):
            w_abs = block_start + w
            for j in range(J):
                for p in range(P):
                    for s_idx in range(S):
                        if solver.Value(sum(x[m, w, j, p, s_idx] for m in range(M))) == 0:
                            continue
                        for m in range(M):
                            if solver.Value(x[m, w, j, p, s_idx]) == 1:
                                # convertit (w_abs,j) -> date approximative avec date_debut
                                d = date_debut + datetime.timedelta(days=w_abs * 7 + j)
                                assignments_all.append({
                                    "date": d.isoformat(),
                                    "slot": "AM" if plages[p] == "M" else "PM",
                                    "day": jours[j],
                                    "room": salles[s_idx],
                                    "doctor": medecins[m].name,
                                    "week_index": w_abs + 1
                                })

    return {"status": "ok", "assignments": assignments_all}