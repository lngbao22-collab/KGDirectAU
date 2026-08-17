import { Fragment, useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { useI18n, type MessageKey } from "../i18n";

export const COLUMN_IDS = ["input", "plot", "info"] as const;
export type ColumnId = (typeof COLUMN_IDS)[number];

const COLUMN_KEYS: Record<ColumnId, MessageKey> = {
  input: "input",
  plot: "plot",
  info: "info",
};

const DEFAULT_WEIGHTS: Record<ColumnId, number> = {
  input: 23,
  plot: 55,
  info: 22,
};

const MIN_PX: Record<ColumnId, number> = {
  input: 200,
  plot: 280,
  info: 200,
};

type Visibility = Record<ColumnId, boolean>;

function visibleIds(visibility: Visibility): ColumnId[] {
  return COLUMN_IDS.filter((id) => visibility[id]);
}

export function ColumnToggles({
  visibility,
  onToggle,
}: {
  visibility: Visibility;
  onToggle: (id: ColumnId) => void;
}) {
  const { t } = useI18n();
  const visibleCount = visibleIds(visibility).length;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{t("columns")}</span>
      {COLUMN_IDS.map((id) => {
        const active = visibility[id];
        const locked = active && visibleCount === 1;
        const label = t(COLUMN_KEYS[id]);
        return (
          <button
            className={`rounded-lg border px-2 py-1 text-xs font-medium ${
              active
                ? "border-brand-600 bg-brand-50 text-brand-700"
                : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50"
            } ${locked ? "cursor-not-allowed opacity-60" : ""}`}
            disabled={locked}
            key={id}
            onClick={() => onToggle(id)}
            title={locked ? t("keepOneColumn") : active ? t("hideColumn", { label }) : t("showColumn", { label })}
            type="button"
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

export function useColumnLayout() {
  const [visibility, setVisibility] = useState<Visibility>({
    input: true,
    plot: true,
    info: true,
  });
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS);

  const toggle = useCallback((id: ColumnId) => {
    setVisibility((current) => {
      const next = { ...current, [id]: !current[id] };
      if (visibleIds(next).length === 0) return current;
      return next;
    });
  }, []);

  return { visibility, weights, setWeights, toggle };
}

type Props = {
  columns: Record<ColumnId, ReactNode>;
  visibility: Visibility;
  weights: Record<ColumnId, number>;
  onWeightsChange: (next: Record<ColumnId, number>) => void;
  onToggle: (id: ColumnId) => void;
};

export function ColumnLayout({ columns, visibility, weights, onWeightsChange, onToggle }: Props) {
  const { t } = useI18n();
  const containerRef = useRef<HTMLDivElement>(null);
  const [wide, setWide] = useState(() =>
    typeof window === "undefined" ? true : window.matchMedia("(min-width: 1280px)").matches,
  );
  const dragRef = useRef<{
    left: ColumnId;
    right: ColumnId;
    startX: number;
    startWeights: Record<ColumnId, number>;
    width: number;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const shown = visibleIds(visibility);

  useEffect(() => {
    const media = window.matchMedia("(min-width: 1280px)");
    const sync = () => setWide(media.matches);
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    return () => window.cancelAnimationFrame(frame);
  }, [visibility]);

  useEffect(() => {
    function onMove(event: PointerEvent) {
      const drag = dragRef.current;
      if (!drag) return;
      const total = shown.reduce((sum, id) => sum + drag.startWeights[id], 0);
      const pxPerWeight = drag.width / total;
      if (!pxPerWeight) return;
      const delta = (event.clientX - drag.startX) / pxPerWeight;
      const leftMin = MIN_PX[drag.left] / pxPerWeight;
      const rightMin = MIN_PX[drag.right] / pxPerWeight;
      const pair = drag.startWeights[drag.left] + drag.startWeights[drag.right];
      const nextLeft = Math.min(pair - rightMin, Math.max(leftMin, drag.startWeights[drag.left] + delta));
      onWeightsChange({
        ...drag.startWeights,
        [drag.left]: nextLeft,
        [drag.right]: pair - nextLeft,
      });
    }
    function onUp() {
      dragRef.current = null;
      setDragging(false);
      document.body.style.removeProperty("cursor");
      document.body.style.removeProperty("user-select");
      window.dispatchEvent(new Event("resize"));
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [onWeightsChange, shown]);

  function startDrag(left: ColumnId, right: ColumnId, event: ReactPointerEvent<HTMLDivElement>) {
    const width = containerRef.current?.getBoundingClientRect().width || 0;
    dragRef.current = {
      left,
      right,
      startX: event.clientX,
      startWeights: { ...weights },
      width,
    };
    setDragging(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    event.preventDefault();
  }

  return (
    <div
      className={`min-w-0 ${wide ? "flex items-stretch" : "grid grid-cols-1 gap-4"} ${dragging ? "select-none" : ""}`}
      ref={containerRef}
    >
      {shown.map((id, index) => {
        const next = shown[index + 1];
        return (
          <Fragment key={id}>
            <div
              className="flex min-w-0 flex-col"
              style={
                wide
                  ? {
                      flexGrow: weights[id],
                      flexShrink: 1,
                      flexBasis: 0,
                      minWidth: MIN_PX[id],
                    }
                  : undefined
              }
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
                  {t(COLUMN_KEYS[id])}
                </span>
                <button
                  className="rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 hover:bg-slate-100 hover:text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                  disabled={shown.length === 1}
                  onClick={() => onToggle(id)}
                  type="button"
                >
                  {t("hide")}
                </button>
              </div>
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">{columns[id]}</div>
            </div>
            {wide && next ? (
              <div
                aria-label={t("resizeColumns", { left: t(COLUMN_KEYS[id]), right: t(COLUMN_KEYS[next]) })}
                className={`group relative w-2 shrink-0 cursor-col-resize ${dragging ? "bg-brand-200" : "hover:bg-brand-100"}`}
                onPointerDown={(event) => startDrag(id, next, event)}
                role="separator"
              >
                <span className="absolute inset-y-8 left-1/2 w-px -translate-x-1/2 bg-slate-200 group-hover:bg-brand-400" />
              </div>
            ) : null}
          </Fragment>
        );
      })}
    </div>
  );
}
