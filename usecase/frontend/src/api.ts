import type { DiseaseDetail, ModelList, PredictResponse, Symptom } from "./types";
import type { Lang } from "./i18n";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      /* keep status text */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

function withLang(path: string, lang: Lang, extra?: Record<string, string>): string {
  const params = new URLSearchParams({ lang, ...(extra || {}) });
  return `${path}?${params.toString()}`;
}

export const api = {
  symptoms: (q = "", lang: Lang = "en") =>
    request<Symptom[]>(withLang("/api/symptoms", lang, q ? { q } : undefined)),
  models: () => request<ModelList>("/api/models"),
  loadModel: (modelId: string) =>
    request<ModelList>(`/api/models/${encodeURIComponent(modelId)}/load`, { method: "POST" }),
  predict: (body: { symptom_ids: string[]; model_id?: string; projection: "umap" | "tsne"; lang: Lang }) =>
    request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify(body) }),
  disease: (diseaseId: string, symptomIds: string[] = [], lang: Lang = "en") =>
    request<DiseaseDetail>(
      withLang(`/api/diseases/${encodeURIComponent(diseaseId)}`, lang, {
        ...(symptomIds.length ? { symptoms: symptomIds.join(",") } : {}),
      }),
    ),
};
