import pandas as pd

# Cache global pour éviter de recharger le fichier à chaque appel
_COST_CACHE = {}

# Nom du fichier de coûts (une feuille par médecin)
COST_FILE = "couts_medecins.xlsx"

def _load_costs():
    """
    Charge le fichier Excel des coûts dans _COST_CACHE.
    La structure en mémoire sera :
        _COST_CACHE[medecin][salle][jour_plage] = valeur
    """
    global _COST_CACHE
    xls = pd.ExcelFile(COST_FILE)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        _COST_CACHE[sheet] = df.to_dict(orient="index")


def cost(medecin, jour, plage, salle):
    global _COST_CACHE

    if not _COST_CACHE:
        _load_costs()

    jour_plage = f"{jour} {plage}"

    try:
        return _COST_CACHE[medecin][salle][jour_plage]
    except KeyError: # éviter les erreur dans le excel
        return 1

