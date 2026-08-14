import { useMemo, useState } from "react";
import { frequencyColor, frequencyLabel } from "../lib/colors";
import { useI18n } from "../i18n";
import type { Candidate } from "../types";

type SortKey = "frequency" | "avg_similarity" | "max_similarity" | "name";

type Props = {
  candidates: Candidate[];
  selectedCount: number;
  focusedId?: string | null;
  onSelect: (id: string) => void;
};

export function RankingTable({ candidates, selectedCount, focusedId, onSelect }: Props) {
  const { lang, t } = useI18n();
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("frequency");
  const [page, setPage] = useState(0);
  const pageSize = 10;

  const rows = useMemo(() => {
    const filtered = candidates.filter((item) =>
      item.name.toLowerCase().includes(query.toLowerCase()) || item.id.toLowerCase().includes(query.toLowerCase()),
    );
    const sorted = [...filtered].sort((a, b) => {
      if (sortKey === "name") return a.name.localeCompare(b.name, lang === "vi" ? "vi" : "en");
      if (sortKey === "frequency") return b.frequency - a.frequency || b.avg_similarity - a.avg_similarity;
      if (sortKey === "max_similarity") return b.max_similarity - a.max_similarity;
      return b.avg_similarity - a.avg_similarity;
    });
    return sorted;
  }, [candidates, lang, query, sortKey]);

  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const current = rows.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <section className="card p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-wide text-slate-500">
          {t("candidateDiseases", { n: candidates.length })}
        </h2>
        <div className="flex items-center gap-2">
          <input
            className="rounded-xl border border-slate-200 px-3 py-1.5 text-sm"
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
            placeholder={t("search")}
            value={query}
          />
          <button
            className="rounded-xl border border-slate-200 px-3 py-1.5 text-sm"
            onClick={() => {
              const blob = new Blob([JSON.stringify(candidates, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = "kgau-candidate-diseases.json";
              link.click();
              URL.revokeObjectURL(url);
            }}
            type="button"
          >
            {t("download")}
          </button>
          <select
            className="rounded-xl border border-slate-200 px-3 py-1.5 text-sm"
            onChange={(event) => setSortKey(event.target.value as SortKey)}
            value={sortKey}
          >
            <option value="frequency">{t("sortFrequency")}</option>
            <option value="avg_similarity">{t("sortAvg")}</option>
            <option value="max_similarity">{t("sortMax")}</option>
            <option value="name">{t("sortName")}</option>
          </select>
        </div>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-400">
            <tr>
              <th className="px-2 py-2">{t("rank")}</th>
              <th className="px-2 py-2">{t("disease")}</th>
              <th className="px-2 py-2">{t("matchedFreq")}</th>
              <th className="px-2 py-2">{t("matchedNames")}</th>
              <th className="px-2 py-2">{t("avgSimilarity")}</th>
              <th className="px-2 py-2">{t("maxSimilarity")}</th>
            </tr>
          </thead>
          <tbody>
            {current.map((item) => (
              <tr
                className={`cursor-pointer border-t border-slate-100 hover:bg-slate-50 ${
                  item.id === focusedId ? "bg-blue-50" : ""
                }`}
                key={item.id}
                onClick={() => onSelect(item.id)}
              >
                <td className="px-2 py-2 font-medium">{item.rank}</td>
                <td className="px-2 py-2">
                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: frequencyColor(item.frequency) }} />
                  {item.name}
                </td>
                <td className="px-2 py-2">
                  <span
                    className="rounded-md px-2 py-0.5 text-xs font-semibold text-white"
                    style={{ background: frequencyColor(item.frequency) }}
                  >
                    {frequencyLabel(item.frequency, selectedCount)}
                  </span>
                </td>
                <td className="px-2 py-2">
                  <div className="flex flex-wrap gap-1">
                    {item.matched_symptoms.map((symptom) => (
                      <button
                        className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 hover:bg-emerald-100"
                        key={symptom.id}
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelect(symptom.id);
                        }}
                        type="button"
                      >
                        {symptom.name}
                      </button>
                    ))}
                  </div>
                </td>
                <td className="px-2 py-2">{item.avg_similarity.toFixed(3)}</td>
                <td className="px-2 py-2">{item.max_similarity.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
        <span>
          {t("page", { page: page + 1, pageCount })}
        </span>
        <div className="flex gap-2">
          <button className="rounded-lg border px-2 py-1" disabled={page === 0} onClick={() => setPage((value) => value - 1)} type="button">
            {t("prev")}
          </button>
          <button
            className="rounded-lg border px-2 py-1"
            disabled={page >= pageCount - 1}
            onClick={() => setPage((value) => value + 1)}
            type="button"
          >
            {t("next")}
          </button>
        </div>
      </div>
    </section>
  );
}
