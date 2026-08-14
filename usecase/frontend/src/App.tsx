import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { ColumnLayout, ColumnToggles, useColumnLayout } from "./components/ColumnLayout";
import { DiseasePanel } from "./components/DiseasePanel";
import { EmbeddingPlot } from "./components/EmbeddingPlot";
import { RankingTable } from "./components/RankingTable";
import { SettingsPanel } from "./components/SettingsPanel";
import { SymptomPanel } from "./components/SymptomPanel";
import { LanguageToggle, useI18n } from "./i18n";
import { DEFAULT_PALETTE_ID, getPalette } from "./lib/colors";
import type { DiseaseDetail, ModelList, PredictResponse, Symptom } from "./types";

export default function App() {
  const { lang, t } = useI18n();
  const [symptoms, setSymptoms] = useState<Symptom[]>([]);
  const [models, setModels] = useState<ModelList | null>(null);
  const [projection, setProjection] = useState<"umap" | "tsne">("tsne");
  const [paletteId, setPaletteId] = useState(DEFAULT_PALETTE_ID);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [focused, setFocused] = useState<DiseaseDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const { visibility, weights, setWeights, toggle } = useColumnLayout();
  const palette = useMemo(() => getPalette(paletteId).colors, [paletteId]);

  useEffect(() => {
    api.models().then(setModels).catch((err) => setError(err.message));
  }, []);

  async function predict(nextProjection = projection, nextSymptoms = symptoms, modelId?: string) {
    if (nextSymptoms.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const payload = await api.predict({
        symptom_ids: nextSymptoms.map((item) => item.id),
        model_id: modelId || models?.current_model_id,
        projection: nextProjection,
        lang,
      });
      setResult(payload);
      setFocused(payload.focused);
      setSymptoms((current) => {
        const names = new Map(payload.selected_symptoms.map((item) => [item.id, item.name]));
        return current.map((item) => (names.has(item.id) ? { ...item, name: names.get(item.id) || item.name } : item));
      });
      setModels((current) =>
        current
          ? { ...current, current_model_id: payload.model.id }
          : current,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : t("predictionFailed"));
    } finally {
      setLoading(false);
    }
  }

  async function changeModel(modelId: string) {
    setLoading(true);
    setError("");
    try {
      const next = await api.loadModel(modelId);
      setModels(next);
      if (symptoms.length > 0) {
        await predict(projection, symptoms, modelId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t("loadModelFailed"));
    } finally {
      setLoading(false);
    }
  }

  async function selectDisease(diseaseId: string) {
    try {
      const detail = await api.disease(diseaseId, symptoms.map((item) => item.id), lang);
      setFocused(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("loadDiseaseFailed"));
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function relocalize() {
      try {
        if (symptoms.length > 0) {
          const rows = await api.symptoms("", lang);
          if (cancelled) return;
          const byId = new Map(rows.map((item) => [item.id, item.name]));
          setSymptoms((current) => current.map((item) => ({ ...item, name: byId.get(item.id) || item.name })));
        }
        if (result) {
          await predict(projection, symptoms);
          return;
        }
        if (focused) {
          const detail = await api.disease(focused.id, symptoms.map((item) => item.id), lang);
          if (!cancelled) setFocused(detail);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : t("predictionFailed"));
      }
    }
    relocalize();
    return () => {
      cancelled = true;
    };
    // Relocalize entity names when the UI language changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <h1 className="text-xl font-semibold text-slate-900">{t("title")}</h1>
            <p className="max-w-3xl text-xs text-slate-500">{t("subtitle")}</p>
          </div>
          <div className="flex items-center gap-4">
            <LanguageToggle />
            <ColumnToggles onToggle={toggle} visibility={visibility} />
            {loading ? <span className="text-xs font-medium text-brand-600">{t("working")}</span> : null}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1600px] space-y-4 px-6 py-4">
        <ColumnLayout
          columns={{
            input: (
              <div className="min-w-0 space-y-4">
                <SymptomPanel
                  loading={loading}
                  onChange={setSymptoms}
                  onClear={() => {
                    setSymptoms([]);
                    setResult(null);
                    setFocused(null);
                  }}
                  onPredict={() => predict()}
                  onSelect={selectDisease}
                  palette={palette}
                  selected={symptoms}
                />
                <SettingsPanel
                  models={models}
                  onModelChange={changeModel}
                  onPaletteChange={setPaletteId}
                  onProjectionChange={(value) => {
                    setProjection(value);
                    if (result) predict(value);
                  }}
                  paletteId={paletteId}
                  projection={projection}
                />
              </div>
            ),
            plot: (
              <div className="flex h-full min-h-0 min-w-0 flex-col">
                <EmbeddingPlot
                  focusedId={focused?.id}
                  onSelect={selectDisease}
                  palette={palette}
                  points={result?.points || []}
                  projection={projection}
                />
              </div>
            ),
            info: (
              <div className="flex h-full min-h-0 min-w-0 flex-col">
                <DiseasePanel detail={focused} onSelectSimilar={selectDisease} />
              </div>
            ),
          }}
          onToggle={toggle}
          onWeightsChange={setWeights}
          visibility={visibility}
          weights={weights}
        />
        <div>
          {error ? <div className="mb-3 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}
          <RankingTable
            candidates={result?.candidates || []}
            focusedId={focused?.id}
            onSelect={selectDisease}
            selectedCount={symptoms.length}
          />
        </div>
      </main>
    </div>
  );
}
