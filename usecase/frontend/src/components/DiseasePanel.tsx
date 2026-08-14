import { frequencyColor, frequencyLabel } from "../lib/colors";
import { useI18n } from "../i18n";
import type { DiseaseDetail } from "../types";

type Props = {
  detail: DiseaseDetail | null;
  onSelectSimilar: (id: string) => void;
};

export function DiseasePanel({ detail, onSelectSimilar }: Props) {
  const { t } = useI18n();
  if (!detail) {
    return (
      <section className="card flex h-full min-h-[640px] min-w-0 flex-col p-4 text-sm text-slate-400 xl:min-h-0">
        <h2 className="mb-2 text-sm font-semibold tracking-wide text-slate-500">{t("focusedEntity")}</h2>
        {t("emptyDetail")}
      </section>
    );
  }

  const isSymptom = detail.kind === "symptom" || !detail.id.startsWith("DOID:");

  return (
    <section className="card flex h-full min-h-[640px] min-w-0 flex-col overflow-x-hidden overflow-y-hidden p-4 xl:min-h-0">
      <div className="min-w-0 shrink-0">
        <div className="text-xs uppercase tracking-wide text-slate-400">
          {isSymptom ? t("symptom") : t("focusedEntity")}
        </div>
        <h2 className="break-words text-xl font-semibold text-slate-900">{detail.name}</h2>
        <div className="mt-1 text-xs text-slate-500">{detail.id}</div>
        <p className="mt-3 max-w-full overflow-hidden break-words text-sm leading-6 text-slate-600 line-clamp-4">
          {detail.description}
        </p>
        <div className="mt-2 max-w-full text-xs text-slate-500">
          {detail.wiki_url ? (
            <a
              className="block max-w-full truncate text-brand-600 hover:underline"
              href={detail.wiki_url}
              rel="noopener noreferrer"
              title={detail.wiki_url}
              target="_blank"
            >
              {detail.wiki_url}
            </a>
          ) : (
            t("noWiki")
          )}
        </div>
      </div>
      {isSymptom ? null : (
        <div className="mt-4 flex min-h-0 min-w-0 flex-1 flex-col gap-4">
          <div className="min-w-0 shrink-0">
            <h3 className="mb-2 text-sm font-semibold text-slate-700">
              {t("matchedSymptoms", { label: frequencyLabel(detail.matched_count, detail.selected_count) })}
            </h3>
            <div className="space-y-1">
              {detail.matched_symptoms.map((item) => (
                <button
                  className="flex w-full min-w-0 items-center gap-2 rounded-lg px-1 py-0.5 text-left text-sm hover:bg-slate-50"
                  key={item.id}
                  onClick={() => onSelectSimilar(item.id)}
                  type="button"
                >
                  <span className="text-emerald-600">✓</span>
                  <span className="min-w-0 truncate">{item.name}</span>
                  <span className="shrink-0 text-xs text-slate-400">{item.score.toFixed(3)}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <h3 className="mb-1 text-sm font-semibold text-slate-700">{t("similarDiseases")}</h3>
            <div className="text-xs text-slate-400">{t("graphLookup")}</div>
            <div className="mt-2 max-h-52 min-h-0 flex-1 overflow-y-auto pr-1">
              {detail.similar_diseases.length === 0 ? (
                <div className="text-sm text-slate-400">{t("noResembles")}</div>
              ) : (
                <div className="space-y-1">
                  {detail.similar_diseases.map((item) => (
                    <button
                      className="block w-full truncate rounded-lg px-2 py-1.5 text-left text-sm hover:bg-slate-50"
                      key={item.id}
                      onClick={() => onSelectSimilar(item.id)}
                      title={item.name}
                      type="button"
                    >
                      {item.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div
            className="h-1.5 shrink-0 rounded-full"
            style={{ background: frequencyColor(detail.matched_count || 1) }}
          />
        </div>
      )}
    </section>
  );
}
