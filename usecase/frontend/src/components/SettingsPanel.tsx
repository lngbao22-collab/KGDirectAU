import { COLOR_PALETTES } from "../lib/colors";
import { useI18n, type MessageKey } from "../i18n";
import type { ModelList } from "../types";

type Props = {
  models: ModelList | null;
  projection: "umap" | "tsne";
  paletteId: string;
  onModelChange: (modelId: string) => void;
  onProjectionChange: (value: "umap" | "tsne") => void;
  onPaletteChange: (paletteId: string) => void;
};

const PALETTE_KEYS: Record<string, MessageKey> = {
  vivid: "palette_vivid",
  colorblind: "palette_colorblind",
  tableau: "palette_tableau",
  ocean: "palette_ocean",
  jewel: "palette_jewel",
  "high-contrast": "palette_highContrast",
};

export function SettingsPanel({
  models,
  projection,
  paletteId,
  onModelChange,
  onProjectionChange,
  onPaletteChange,
}: Props) {
  const { t } = useI18n();
  return (
    <section className="card space-y-4 p-4">
      <h2 className="text-sm font-semibold tracking-wide text-slate-500">{t("advancedSettings")}</h2>
      <label className="block text-xs font-medium text-slate-500">
        {t("backboneModel")}
        <select
          className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
          onChange={(event) => onModelChange(event.target.value)}
          value={models?.current_model_id || ""}
        >
          {(models?.models || []).map((item) => (
            <option disabled={!item.available} key={item.id} value={item.id}>
              {item.label.replace(" (default)", ` ${t("defaultSuffix")}`)}
              {item.available ? "" : ` ${t("unavailable")}`}
            </option>
          ))}
        </select>
      </label>
      <div>
        <div className="mb-1 text-xs font-medium text-slate-500">{t("embeddingProjection")}</div>
        <div className="flex gap-2">
          {(["tsne", "umap"] as const).map((item) => (
            <button
              className={`flex-1 whitespace-nowrap rounded-xl px-3 py-2 text-sm ${
                projection === item ? "bg-brand-600 text-white" : "border border-slate-200 text-slate-600"
              }`}
              key={item}
              onClick={() => onProjectionChange(item)}
              type="button"
            >
              {item === "umap" ? t("umap") : t("tsne")}
            </button>
          ))}
        </div>
      </div>
      <div>
        <div className="mb-1 text-xs font-medium text-slate-500">{t("symptomColorCombo")}</div>
        <div className="space-y-1.5">
          {COLOR_PALETTES.map((palette) => {
            const selected = palette.id === paletteId;
            return (
              <button
                className={`flex w-full items-center justify-between gap-2 rounded-xl border px-2 py-1.5 text-left ${
                  selected ? "border-brand-600 bg-brand-50" : "border-slate-200 hover:bg-slate-50"
                }`}
                key={palette.id}
                onClick={() => onPaletteChange(palette.id)}
                type="button"
              >
                <span className={`text-xs font-medium ${selected ? "text-brand-700" : "text-slate-600"}`}>
                  {t(PALETTE_KEYS[palette.id] || "palette_vivid")}
                </span>
                <span className="flex shrink-0 gap-1">
                  {palette.colors.map((color) => (
                    <span
                      className="h-3.5 w-3.5 rounded-full"
                      key={color.fill}
                      style={{ background: color.fill, boxShadow: `0 0 0 1px ${color.stroke}` }}
                    />
                  ))}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
