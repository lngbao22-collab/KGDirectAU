import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n";
import { symptomColor, type SymptomStyle } from "../lib/colors";
import type { Symptom } from "../types";

type Props = {
  selected: Symptom[];
  onChange: (next: Symptom[]) => void;
  onPredict: () => void;
  onClear: () => void;
  onSelect?: (id: string) => void;
  loading: boolean;
  palette: SymptomStyle[];
};

export function SymptomPanel({ selected, onChange, onPredict, onClear, onSelect, loading, palette }: Props) {
  const { lang, t } = useI18n();
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState<Symptom[]>([]);
  const [open, setOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      api.symptoms(query, lang).then(setOptions).catch(() => setOptions([]));
    }, 120);
    return () => window.clearTimeout(handle);
  }, [lang, query]);

  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, []);

  const available = useMemo(
    () => options.filter((item) => !selected.some((picked) => picked.id === item.id)),
    [options, selected],
  );

  function addSymptom(item: Symptom) {
    if (selected.length >= 5) return;
    onChange([...selected, item]);
    setQuery("");
    setOpen(false);
  }

  return (
    <section className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-slate-500">{t("inputSymptoms")}</h2>
        <span className="text-xs text-slate-400">{selected.length}/5</span>
      </div>
      <div className="space-y-2">
        {selected.map((item) => (
          <div key={item.id} className="flex items-center justify-between rounded-xl bg-orange-50 px-3 py-2">
            <button
              className="flex min-w-0 items-center gap-2 text-left"
              onClick={() => onSelect?.(item.id)}
              type="button"
            >
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: symptomColor(selected.findIndex((row) => row.id === item.id), palette) }}
              />
              <span className="truncate text-sm font-medium text-slate-800">{item.name}</span>
            </button>
            <button
              className="text-slate-400 hover:text-slate-700"
              onClick={() => onChange(selected.filter((row) => row.id !== item.id))}
              type="button"
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <div className="relative mt-3" ref={searchRef}>
        <input
          autoComplete="off"
          className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none ring-brand-500 focus:ring-2"
          disabled={selected.length >= 5}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          placeholder={selected.length >= 5 ? t("maxSymptoms") : t("addSymptom")}
          value={query}
        />
        {open && available.length > 0 && selected.length < 5 && (
          <div className="absolute z-20 mt-1 max-h-80 w-full overflow-auto rounded-xl border border-slate-200 bg-white shadow-card">
            {available.map((item) => (
              <button
                className="block w-full px-3 py-2 text-left text-sm hover:bg-slate-50"
                key={item.id}
                onClick={() => addSymptom(item)}
                type="button"
              >
                <div className="font-medium">{item.name}</div>
                <div className="text-xs text-slate-400">{item.id}</div>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="mt-3 flex gap-2">
        <button
          className="rounded-xl bg-brand-600 px-3 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          disabled={selected.length === 0 || loading}
          onClick={onPredict}
          type="button"
        >
          {loading ? t("predicting") : t("predict")}
        </button>
        <button
          className="rounded-xl border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50"
          onClick={onClear}
          type="button"
        >
          {t("clear")}
        </button>
      </div>
    </section>
  );
}
