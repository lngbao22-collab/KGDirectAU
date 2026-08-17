from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
USECASE_DIR = BACKEND_DIR.parent
REPO_ROOT = USECASE_DIR.parent
DATABASE_DIR = USECASE_DIR / "database" / "hetionet_subset"
PREPROCESSED_DIR = DATABASE_DIR / "preprocessed"
MODEL_DIR = USECASE_DIR / "model"
FRONTEND_DIR = USECASE_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"

ENTITIES_JSON = PREPROCESSED_DIR / "entities.json"
RELATION2ID_JSON = PREPROCESSED_DIR / "relation2id.json"
DISEASE_METADATA = DATABASE_DIR / "disease_metadata.tsv"
SYMPTOM_METADATA = DATABASE_DIR / "symptom_metadata.tsv"
DISEASE_METADATA_VN = DATABASE_DIR / "disease_metadata_vn.tsv"
SYMPTOM_METADATA_VN = DATABASE_DIR / "symptom_metadata_vn.tsv"
RESEMBLE_JSON = DATABASE_DIR / "resemble_diseases.json"
GROUND_TRUTH = DATABASE_DIR / "ground_truth.txt"
REGISTRY_JSON = MODEL_DIR / "registry.json"
