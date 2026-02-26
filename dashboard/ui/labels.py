TAB_LABELS_MAIN = ["Equipe", "Etude Joueurs", "Ligue", "Classement"]

LABEL_ALL_SEASONS = "\u2014 Toutes \u2014"
LABEL_ALL_PLAYERS = "\u2014 Tous les joueurs \u2014"
LABEL_NO_PLAYERS = "\u2014 Aucun joueur disponible \u2014"

STUDY_SUBTAB_LEADERS = "Leaders"

STUDY_TITLE = "Etude Joueurs (FBref - saisons completes)"
STUDY_SECTION_SELECTION = "### Selection de l'etude"
STUDY_SECTION_VIEWS = "### Vues"

STUDY_INFO_MISSING_DATA = (
    "Donnees d'etude FBref non generees. Lance `make study-fbref` (scraping direct) "
    "ou utilise le mode manuel CSV (`FBREF_STUDY_SOURCE=manual_csv`) puis rebuild le dashboard."
)
STUDY_INFO_MANUAL_MODE = (
    "Mode manuel: depose `data/study/fbref_input/player_match_manual.csv` "
    "puis lance `make study-fbref-manual-docker`."
)
STUDY_WARNING_EMPTY_FILES = "Les fichiers FBref sont presents mais vides. Verifie l'extraction."
STUDY_INFO_REGULARITY_UNAVAILABLE = (
    "Regularite match-par-match indisponible avec la source actuelle (CSV FBref standard saison). "
    "Les vues Progression / Performance restent disponibles."
)
STUDY_INFO_SELECT_PLAYER_TEMPLATE = (
    "Selectionne un joueur pour afficher la fiche detaillee, ou garde '{all_players}' pour les vues globales."
)
STUDY_INFO_PLAYER_NO_REGULARITY = (
    "Aucune donnee de regularite pour ce joueur (souvent seuil de minutes non atteint)."
)
