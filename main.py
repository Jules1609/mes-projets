from ortools.sat.python import cp_model
import pandas as pd
import datetime
from data import cost


class Medecin:
    def __init__(self, name, max_jours, jours_interdits=None, salles_autorisees=None, vacances=None):
        self.name = name
        self.max_jours = max_jours
        self.jours_interdits = jours_interdits or []
        self.salles_autorisees = salles_autorisees or []
        self.vacances = vacances or []


# Le fichier contient : name, max_jours, jours_interdits, salles_autorisees, vacances
# jours_interdits : "Jour:Plage;Jour:Plage" 
# salles_autorisees : "Salle1;Salle2"
# vacances : "dd/mm/YYYY;dd/mm/YYYY"

df = pd.read_excel("medecins.xlsx")

medecins = []
for _, row in df.iterrows():

    
    ji_raw = row.get("jours_interdits", "")
    jours_interdits = []
    if isinstance(ji_raw, str) and ji_raw.strip():
        for pair in ji_raw.split(";"):
            jour, plage = pair.split(":")
            jours_interdits.append((jour.strip(), plage.strip()))

    
    sa_raw = row.get("salles_autorisees", "")
    salles_autorisees = []
    if isinstance(sa_raw, str) and sa_raw.strip():
        salles_autorisees = [s.strip() for s in sa_raw.split(";")]

    
    vac_raw = row.get("vacances", "")
    vacances = []
    if isinstance(vac_raw, str) and vac_raw.strip():
        for d in vac_raw.split(";"):
            d = d.strip()
            if d:
                vacances.append(datetime.datetime.strptime(d, "%d/%m/%Y").date())

    medecins.append(
        Medecin(
            name=row["name"],
            max_jours=int(row["max_jours"]),
            jours_interdits=jours_interdits,
            salles_autorisees=salles_autorisees,
            vacances=vacances,
        )
    )



# Définition des jours, plages
jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
plages = ["M", "A"]

# Le fichier contient : salle, jours_disponibles
# jours_disponibles : "Lundi Matin;Mardi Après-midi;..."
df_salles = pd.read_excel("salles.xlsx")
salles = df_salles["salle"].tolist()

disponibilites_salles = {}
for _, row in df_salles.iterrows():
    dispo_raw = str(row.get("jours_disponibles", "") or "")
    dispo_set = set()
    if dispo_raw.strip():
        for token in dispo_raw.split(";"):
            dispo_set.add(token.strip())
    disponibilites_salles[row["salle"]] = dispo_set

semaines = list(range(8))  # nombre de salles du planning

M = len(medecins)
J = len(jours)
P = len(plages)
S = len(salles)
W = len(semaines)

model = cp_model.CpModel()


## Variables
# x[m][w][j][p][s] = 1 si médecin m travaille semaine w, jour j, plage p, salle s

x = {}
for m in range(M):
    for w in range(W):
        for j in range(J):
            for p in range(P):
                for s in range(S):
                    x[m, w, j, p, s] = model.NewBoolVar(f"x_{m}_{w}_{j}_{p}_{s}")



## Contraintes





# 1. 1 médecin par salle par plage pour chaque semaine,par la salle est ouverte
for w in range(W):
    for j in range(J):
        for p in range(P):
            for s in range(S):
                jour_plage = f"{jours[j]} {plages[p]}"
                salle_nom = salles[s]
                if jour_plage in disponibilites_salles.get(salle_nom, set()):
                    # Salle ouverte : 1 médecin
                    model.Add(sum(x[m, w, j, p, s] for m in range(M)) == 1)
                else:
                    # Salle fermée : aucun médecin 
                    model.Add(sum(x[m, w, j, p, s] for m in range(M)) == 0)


# 2. pas plus d'un médecin par salle
for m in range(M):
    for w in range(W):
        for j in range(J):
            for p in range(P):
                model.Add(sum(x[m, w, j, p, s] for s in range(S)) <= 1)
   


# 3. Jours/plages interdits pour chaque médecin
for m in range(M):
    for (jour_interdit, plage_interdite) in medecins[m].jours_interdits:
        if jour_interdit in jours and plage_interdite in plages:
            j_idx = jours.index(jour_interdit)
            p_idx = plages.index(plage_interdite)
            for w in range(W):
                for s in range(S):
                    model.Add(x[m, w, j_idx, p_idx, s] == 0)

# 4. Contraintes de vacances

DATE_DEBUT = datetime.date(2025, 1, 6)

vacances_indices = {m: [] for m in range(M)}
for m in range(M):
    for d in medecins[m].vacances:
        delta = (d - DATE_DEBUT).days
        if 0 <= delta < W * 7:
            semaine_idx = delta // 7
            jour_semaine = delta % 7
            if jour_semaine < J:
                vacances_indices[m].append((semaine_idx, jour_semaine))

for m in range(M):
    for (w_idx, j_idx) in vacances_indices[m]:
        for p in range(P):
            for s in range(S):
                model.Add(x[m, w_idx, j_idx, p, s] == 0)

# 5. Chaque médecin ne doit pas dépasser son nombre max de jours par semaine
for m in range(M):
    for w in range(W):
        model.Add(
            sum(
                x[m, w, j, p, s]
                for j in range(J)
                for p in range(P)
                for s in range(S)
            )
            <= 2*medecins[m].max_jours
        )

# 6. Restriction par salle (spécialité + disponibilités des salles)
for m in range(M):
    for w in range(W):
        for j in range(J):
            for p in range(P):
                jour_plage = f"{jours[j]} {plages[p]}"
                for s in range(S):
                    salle_nom = salles[s]
                    if (
                        salle_nom not in medecins[m].salles_autorisees
                        or jour_plage not in disponibilites_salles.get(salle_nom, set())
                    ):
                        model.Add(x[m, w, j, p, s] == 0)



# Fonction objectif : minimser coût total + équité entre médecins


alpha = 1  # poids normal

# 1. Calcul du coût total par médecin
cout_total = []
for m in range(M):
    cout_total_m = sum(
        cost(medecins[m].name, jours[j], plages[p], salles[s]) * x[m, w, j, p, s]
        for w in range(W)
        for j in range(J)
        for p in range(P)
        for s in range(S)
    )
    cout_total.append(cout_total_m)

# 2. Nombre de jours travaillés par médecin (jour compté si au moins une plage est travaillée)
jours_travailles = []
for m in range(M):
    total_days = []
    for w in range(W):
        for j in range(J):
            b = model.NewBoolVar(f"jour_work_{m}_{w}_{j}")
            model.Add(sum(x[m, w, j, p, s] for p in range(P) for s in range(S)) >= 1).OnlyEnforceIf(b)
            model.Add(sum(x[m, w, j, p, s] for p in range(P) for s in range(S)) == 0).OnlyEnforceIf(b.Not())
            total_days.append(b)
    jours_travailles.append(sum(total_days))


# Objectif pondéré : coût total + équité entre médecins
alpha = 100 # poids coût total
beta = 1 # poids équité

# Équité : différences absolues de coût total
abs_diff_terms = []
for m1 in range(M):
    for m2 in range(m1 + 1, M):
        diff = model.NewIntVar(-10_000_000, 10_000_000, f"diff_costtotal_{m1}_{m2}")
        absdiff = model.NewIntVar(0, 10_000_000, f"absdiff_costtotal_{m1}_{m2}")
        model.Add(diff == cout_total[m1] - cout_total[m2])
        model.AddAbsEquality(absdiff, diff)
        abs_diff_terms.append(absdiff)

# Objectif : minimiser alpha * coût total + beta * inéquité
model.Minimize(alpha * sum(cout_total) + beta * sum(abs_diff_terms))


## Résolution

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 5  # limite de sécurité
status = solver.Solve(model)


## Affichage du résultat

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("\n===== PLANNING MÉDICAL =====\n")
    for w in range(W):
        print(f"===== Semaine {w+1} =====")
        for j in range(J):
            print(f"--- {jours[j]} ---")
            for p in range(P):
                for s in range(S):
                    print(f"  {plages[p]} - {salles[s]} : ", end="")
                    present = []
                    for m in range(M):
                        if solver.Value(x[m, w, j, p, s]) == 1:
                            present.append(medecins[m].name)
                    print(", ".join(present))
        print()

    # Export Excel : une feuille par semaine, colonnes = jour+plage, lignes = salles
    with pd.ExcelWriter("planning.xlsx") as writer:
        for w in range(W):
            columns = [f"{jours[j]} {plages[p]}" for j in range(J) for p in range(P)]
            result = {salle: {col: "" for col in columns} for salle in salles}

            for j in range(J):
                for p in range(P):
                    col = f"{jours[j]} {plages[p]}"
                    for s in range(S):
                        for m in range(M):
                            if solver.Value(x[m, w, j, p, s]) == 1:
                                result[salles[s]][col] = medecins[m].name

            df_out = pd.DataFrame(result).T
            df_out.to_excel(writer, sheet_name=f"Semaine {w+1}")

    
    # Export Excel par médecin : planning_par_medecin.xlsx
    
    with pd.ExcelWriter("planning_par_medecin.xlsx") as writer2:
        for m in range(M):
            rows = []
            for w in range(W):
                for j in range(J):
                    for p in range(P):
                        for s in range(S):
                            if solver.Value(x[m, w, j, p, s]) == 1:
                                rows.append({
                                    "Semaine": w + 1,
                                    "Jour": jours[j],
                                    "Plage": plages[p],
                                    "Salle": salles[s]
                                })
            df_med = pd.DataFrame(rows)
            df_med.to_excel(writer2, sheet_name=medecins[m].name, index=False)

    print("Fichier Excel généré : planning_par_medecin.xlsx (1 feuille par médecin)")
    print("\nFichier Excel généré : planning.xlsx (8 semaines, format jour+plage colonnes, salles lignes)")
else:
    print("⚠️ Pas de solution trouvée.")