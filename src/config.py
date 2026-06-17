"""Configurações globais do projeto GestRisk CCF."""

from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reprodutibilidade
RANDOM_STATE = 42
N_SAMPLES = 5000
TEST_SIZE = 0.20
N_FOLDS = 10

# Prevalência alvo da classe positiva (3%–8%)
TARGET_PREVALENCE_MIN = 0.03
TARGET_PREVALENCE_MAX = 0.08

# Grupos de variáveis por experimento
FEATURE_GROUPS = {
    "maternal": [
        "idade_materna",
        "etnia",
        "imc_pre_gestacional",
        "paridade",
        "abortamentos_previos",
        "historia_natimorto",
        "filho_anterior_cardiopatia",
        "historia_familiar_cardiopatia",
    ],
    "doencas": [
        "diabetes_pre_gestacional",
        "diabetes_gestacional",
        "lupus",
        "fenilcetonuria",
        "doenca_tireoide",
        "hipertensao_cronica",
    ],
    "infeccoes": [
        "rubeola",
        "citomegalovirus",
        "toxoplasmose",
        "sifilis",
    ],
    "medicacoes": [
        "isotretinoina",
        "acido_valproico",
        "litio",
    ],
    "habitos": [
        "alcool",
        "tabagismo",
        "drogas_ilicitas",
    ],
    "exposicao": [
        "solventes",
        "pesticidas",
    ],
    "gestacionais": [
        "idade_gestacional",
        "gestacao_gemelar",
        "corionicidade",
    ],
    "marcadores_fetais": [
        "translucencia_nucal_aumentada",
        "alteracao_ducto_venoso",
        "regurgitacao_tricuspide",
        "malformacao_extracardiaca",
        "restricao_crescimento_fetal",
        "polidramnio",
        "oligodramnio",
    ],
}

EXPERIMENTS = {
    "experimento_1_maternal": FEATURE_GROUPS["maternal"],
    "experimento_2_maternal_doencas": (
        FEATURE_GROUPS["maternal"]
        + FEATURE_GROUPS["doencas"]
        + FEATURE_GROUPS["infeccoes"]
        + FEATURE_GROUPS["medicacoes"]
        + FEATURE_GROUPS["habitos"]
        + FEATURE_GROUPS["exposicao"]
        + FEATURE_GROUPS["gestacionais"]
    ),
    "experimento_3_completo": (
        FEATURE_GROUPS["maternal"]
        + FEATURE_GROUPS["doencas"]
        + FEATURE_GROUPS["infeccoes"]
        + FEATURE_GROUPS["medicacoes"]
        + FEATURE_GROUPS["habitos"]
        + FEATURE_GROUPS["exposicao"]
        + FEATURE_GROUPS["gestacionais"]
        + FEATURE_GROUPS["marcadores_fetais"]
    ),
}

TARGET_COL = "cardiopatia_congenita"

CATEGORICAL_COLS = ["etnia", "corionicidade"]
BINARY_COLS = [
    "historia_natimorto",
    "filho_anterior_cardiopatia",
    "historia_familiar_cardiopatia",
    "diabetes_pre_gestacional",
    "diabetes_gestacional",
    "lupus",
    "fenilcetonuria",
    "doenca_tireoide",
    "hipertensao_cronica",
    "rubeola",
    "citomegalovirus",
    "toxoplasmose",
    "sifilis",
    "isotretinoina",
    "acido_valproico",
    "litio",
    "alcool",
    "tabagismo",
    "drogas_ilicitas",
    "solventes",
    "pesticidas",
    "gestacao_gemelar",
    "translucencia_nucal_aumentada",
    "alteracao_ducto_venoso",
    "regurgitacao_tricuspide",
    "malformacao_extracardiaca",
    "restricao_crescimento_fetal",
    "polidramnio",
    "oligodramnio",
]

NUMERIC_COLS = [
    "idade_materna",
    "imc_pre_gestacional",
    "paridade",
    "abortamentos_previos",
    "idade_gestacional",
]

BALANCE_STRATEGIES = [
    "none",
    "smote",
    "adasyn",
    "random_oversampling",
    "class_weight",
]

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Extra Trees",
    "Gradient Boosting",
    "Decision Tree",
    "SVM",
    "KNN",
    "Naive Bayes",
    "MLP Neural Network",
]
