export type Symptom = {
  id: string;
  name: string;
};

export type ModelCard = {
  id: string;
  label: string;
  available: boolean;
  framework: string;
  backbone: string;
  embedding_dim: number | null;
  training_strategy: string;
  negative_sampling: string;
  best_epoch: number | null;
  checkpoint: string;
  tuni: number | null;
  gamma_q: number | null;
  gamma_t: number | null;
  gamma_ent: number | null;
  head_eval_mode: string;
};

export type ModelList = {
  default_model_id: string;
  current_model_id: string;
  models: ModelCard[];
};

export type MatchedSymptom = {
  id: string;
  name: string;
  score: number;
  is_ground_truth?: boolean;
};

export type SymptomMetric = {
  id: string;
  name: string;
  precision: number;
  recall: number;
  true_positives: number;
  predicted_count: number;
  ground_truth_count: number;
  top_k: number;
};

export type Candidate = {
  rank: number;
  id: string;
  name: string;
  frequency: number;
  avg_similarity: number;
  max_similarity: number;
  matched_symptoms: MatchedSymptom[];
  ground_truth_hits?: number;
};

export type ScatterPoint = {
  id: string;
  name: string;
  kind: "symptom" | "disease";
  x: number;
  y: number;
  frequency?: number | null;
  avg_similarity?: number | null;
  rank?: number | null;
  symptom_ids?: string[];
  ground_truth_ids?: string[];
  precision?: number | null;
  recall?: number | null;
  true_positives?: number | null;
  predicted_count?: number | null;
  ground_truth_count?: number | null;
  top_k?: number | null;
};

export type SimilarDisease = {
  id: string;
  name: string;
};

export type DiseaseDetail = {
  id: string;
  name: string;
  kind?: "disease" | "symptom";
  description: string;
  wiki_url: string;
  hetionet_id: string;
  matched_symptoms: MatchedSymptom[];
  matched_count: number;
  selected_count: number;
  similar_diseases: SimilarDisease[];
  precision?: number | null;
  recall?: number | null;
  true_positives?: number | null;
  predicted_count?: number | null;
  ground_truth_count?: number | null;
  top_k?: number | null;
};

export type PredictResponse = {
  model: ModelCard;
  projection: "umap" | "tsne";
  selected_symptoms: Symptom[];
  candidates: Candidate[];
  points: ScatterPoint[];
  focused: DiseaseDetail | null;
  symptom_metrics?: SymptomMetric[];
};
