import { useEffect, useMemo, useRef, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import {
  MIXED_OUTLINE,
  pieFill,
  pieStroke,
  symptomColorMap,
  symptomStrokeMap,
  type SymptomStyle,
} from "../lib/colors";
import { useI18n } from "../i18n";
import type { ScatterPoint, SymptomMetric } from "../types";
import { MetricsHover } from "./MetricsHover";

const Plot = createPlotlyComponent(Plotly);

type Props = {
  points: ScatterPoint[];
  projection: "umap" | "tsne";
  focusedId?: string | null;
  palette: SymptomStyle[];
  onSelect: (id: string) => void;
};

type PieMarker = {
  id: string;
  left: number;
  top: number;
  size: number;
  colors: string[];
  stroke: string;
  strokeWidth: number;
};

type HoverTip = {
  id: string;
  left: number;
  top: number;
};

type PlotHoverPoint = {
  customdata?: unknown;
  bbox?: { x0: number; x1: number; y0: number; y1: number };
};

type PlotHoverEvent = {
  points?: PlotHoverPoint[];
  event?: { offsetX?: number; offsetY?: number };
};

function pointId(customdata: unknown): string | null {
  if (typeof customdata === "string") return customdata;
  if (Array.isArray(customdata) && typeof customdata[0] === "string") return customdata[0];
  return null;
}

function symptomMetric(item: ScatterPoint): SymptomMetric | null {
  if (item.precision == null || item.recall == null) return null;
  return {
    id: item.id,
    name: item.name,
    precision: item.precision,
    recall: item.recall,
    true_positives: item.true_positives ?? 0,
    predicted_count: item.predicted_count ?? 0,
    ground_truth_count: item.ground_truth_count ?? 0,
    top_k: item.top_k ?? 10,
  };
}

function densityScales(points: ScatterPoint[]): number[] {
  const n = points.length;
  if (n === 0) return [];
  const xs = points.map((item) => item.x);
  const ys = points.map((item) => item.y);
  const xSpan = Math.max(...xs) - Math.min(...xs) || 1;
  const ySpan = Math.max(...ys) - Math.min(...ys) || 1;
  const radius2 = (0.028 * xSpan) ** 2 + (0.028 * ySpan) ** 2;
  return points.map((item, index) => {
    let count = 0;
    for (let other = 0; other < n; other += 1) {
      if (other === index) continue;
      const dx = xs[other] - item.x;
      const dy = ys[other] - item.y;
      if (dx * dx + dy * dy < radius2) count += 1;
    }
    return 1 / Math.sqrt(1 + count * 0.28);
  });
}

function markerSize(
  disease: ScatterPoint,
  density: number,
  focusedId?: string | null,
  gtTargets?: Set<string>,
): number {
  const similarity = disease.avg_similarity != null ? disease.avg_similarity : 0.35;
  const base = (8.5 + similarity * 10) * density;
  const size = Math.min(18, Math.max(7, base));
  if (disease.id === focusedId) return size + 3;
  if (gtTargets?.has(disease.id)) return size + 2;
  return size;
}

function samePies(left: PieMarker[], right: PieMarker[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((item, index) => {
    const other = right[index];
    return (
      item.id === other.id &&
      item.left === other.left &&
      item.top === other.top &&
      item.size === other.size &&
      item.stroke === other.stroke &&
      item.strokeWidth === other.strokeWidth &&
      item.colors.join() === other.colors.join()
    );
  });
}

type PlotGraph = HTMLElement & {
  _fullLayout?: {
    xaxis?: { l2p?: (value: number) => number; _offset?: number };
    yaxis?: { l2p?: (value: number) => number; _offset?: number };
  };
};

function pointAnchor(graphDiv: PlotGraph, item: ScatterPoint): { left: number; top: number } | null {
  const xToPx = graphDiv._fullLayout?.xaxis?.l2p;
  const yToPx = graphDiv._fullLayout?.yaxis?.l2p;
  const xOffset = graphDiv._fullLayout?.xaxis?._offset || 0;
  const yOffset = graphDiv._fullLayout?.yaxis?._offset || 0;
  if (!xToPx || !yToPx) return null;
  return {
    left: xToPx(item.x) + xOffset,
    top: yToPx(item.y) + yOffset,
  };
}

function hoverAnchor(point: PlotHoverPoint, event?: PlotHoverEvent["event"]): { left: number; top: number } {
  if (point.bbox) {
    return {
      left: (point.bbox.x0 + point.bbox.x1) / 2,
      top: point.bbox.y0,
    };
  }
  return { left: event?.offsetX ?? 0, top: event?.offsetY ?? 0 };
}

function layoutPies(
  graphDiv: PlotGraph,
  diseases: ScatterPoint[],
  colorMap: Record<string, string>,
  densities: number[],
  palette: SymptomStyle[],
  focusedId?: string | null,
  gtTargets?: Set<string>,
): PieMarker[] {
  const xToPx = graphDiv._fullLayout?.xaxis?.l2p;
  const yToPx = graphDiv._fullLayout?.yaxis?.l2p;
  const xOffset = graphDiv._fullLayout?.xaxis?._offset || 0;
  const yOffset = graphDiv._fullLayout?.yaxis?._offset || 0;
  if (!xToPx || !yToPx) return [];
  return diseases.map((disease, index) => {
    const ids = disease.symptom_ids?.length ? disease.symptom_ids : [];
    const colors = ids.map((id) => colorMap[id] || palette[0].fill);
    const focused = disease.id === focusedId;
    const gt = gtTargets?.has(disease.id) || false;
    return {
      id: disease.id,
      left: xToPx(disease.x) + xOffset,
      top: yToPx(disease.y) + yOffset,
      size: markerSize(disease, densities[index] ?? 1, focusedId, gtTargets),
      colors,
      stroke: pieStroke(colors, palette),
      strokeWidth: focused ? 2 : gt ? 1.6 : 1.2,
    };
  });
}

export function EmbeddingPlot({ points, projection, focusedId, palette, onSelect }: Props) {
  const { t } = useI18n();
  const plotRef = useRef<PlotGraph | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [pies, setPies] = useState<PieMarker[]>([]);
  const [hover, setHover] = useState<HoverTip | null>(null);
  const hoverIdRef = useRef<string | null>(null);
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const symptoms = useMemo(() => points.filter((item) => item.kind === "symptom"), [points]);
  const diseases = useMemo(() => points.filter((item) => item.kind === "disease"), [points]);
  const densities = useMemo(() => densityScales(diseases), [diseases]);
  const colorMap = useMemo(
    () => symptomColorMap(symptoms.map((item) => item.id), palette),
    [palette, symptoms],
  );
  const strokeMap = useMemo(
    () => symptomStrokeMap(symptoms.map((item) => item.id), palette),
    [palette, symptoms],
  );
  const viewRevision = `${projection}:${symptoms.map((item) => item.id).join(",")}`;
  const hoverPoint = hover ? points.find((item) => item.id === hover.id) ?? null : null;
  const gtTargets = useMemo(() => {
    const focused = symptoms.find((item) => item.id === focusedId);
    return new Set(focused?.ground_truth_ids || []);
  }, [focusedId, symptoms]);

  const diseaseTrace = useMemo(
    () => ({
      x: diseases.map((item) => item.x),
      y: diseases.map((item) => item.y),
      text: diseases.map((item) => item.name),
      customdata: diseases.map((item) => [item.id, item.frequency || 0, item.avg_similarity ?? null]),
      mode: "markers" as const,
      type: "scatter" as const,
      name: t("traceDiseases"),
      opacity: 1,
      hoverinfo: "none" as const,
      marker: {
        symbol: "circle",
        size: diseases.map((item, index) => markerSize(item, densities[index] ?? 1, focusedId, gtTargets)),
        color: diseases.map((item) => colorMap[item.symptom_ids?.[0] || ""] || palette[0].fill),
        opacity: 0,
        line: { width: 0 },
      },
    }),
    [colorMap, densities, diseases, focusedId, gtTargets, palette, t],
  );

  const symptomTrace = useMemo(
    () => ({
      x: symptoms.map((item) => item.x),
      y: symptoms.map((item) => item.y),
      text: symptoms.map((item) => item.name),
      customdata: symptoms.map((item) => [item.id]),
      mode: "markers+text" as const,
      type: "scatter" as const,
      name: t("traceSymptoms"),
      opacity: 1,
      textposition: "top center",
      textfont: { size: 11, color: "#0f172a", family: "Segoe UI, system-ui, sans-serif" },
      cliponaxis: false,
      hoverinfo: "none" as const,
      marker: {
        symbol: "triangle-up",
        size: symptoms.map((item) => (item.id === focusedId ? 16 : 14)),
        color: symptoms.map((item) => colorMap[item.id] || palette[0].fill),
        opacity: 1,
        line: {
          width: 1.4,
          color: symptoms.map((item) => strokeMap[item.id] || MIXED_OUTLINE),
        },
      },
    }),
    [colorMap, focusedId, palette, strokeMap, symptoms, t],
  );

  const shapes = useMemo(() => {
    const focused = symptoms.find((item) => item.id === focusedId);
    if (!focused) return [];
    const byId = new Map(diseases.map((item) => [item.id, item]));
    const color = strokeMap[focused.id] || colorMap[focused.id] || MIXED_OUTLINE;
    return (focused.ground_truth_ids || []).flatMap((diseaseId) => {
      const target = byId.get(diseaseId);
      if (!target) return [];
      return [
        {
          type: "line" as const,
          xref: "x" as const,
          yref: "y" as const,
          x0: focused.x,
          y0: focused.y,
          x1: target.x,
          y1: target.y,
          layer: "below" as const,
          opacity: 1,
          line: { color, width: 1.4 },
        },
      ];
    });
  }, [colorMap, diseases, focusedId, strokeMap, symptoms]);

  function syncHoverPosition(graphDiv: PlotGraph) {
    const id = hoverIdRef.current;
    if (!id) {
      setHover((current) => (current ? null : current));
      return;
    }
    const item = pointsRef.current.find((point) => point.id === id);
    if (!item) {
      hoverIdRef.current = null;
      setHover(null);
      return;
    }
    const anchor = pointAnchor(graphDiv, item);
    if (!anchor) return;
    setHover((current) =>
      current && current.id === id && current.left === anchor.left && current.top === anchor.top
        ? current
        : { id, ...anchor },
    );
  }

  function setHovered(id: string | null, fallback?: { left: number; top: number }) {
    hoverIdRef.current = id;
    const graphDiv = plotRef.current;
    if (id && graphDiv) {
      const item = pointsRef.current.find((point) => point.id === id);
      const anchor = item ? pointAnchor(graphDiv, item) : null;
      if (anchor) {
        setHover({ id, ...anchor });
        return;
      }
    }
    if (id && fallback) {
      setHover({ id, ...fallback });
      return;
    }
    setHover(null);
  }

  function syncPies(graphDiv: PlotGraph) {
    plotRef.current = graphDiv;
    const next = layoutPies(graphDiv, diseases, colorMap, densities, palette, focusedId, gtTargets);
    setPies((current) => (samePies(current, next) ? current : next));
    syncHoverPosition(graphDiv);
  }

  useEffect(() => {
    hoverIdRef.current = null;
    setHover(null);
  }, [viewRevision]);

  useEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    let timer = 0;
    const observer = new ResizeObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        if (plotRef.current) Plotly.Plots.resize(plotRef.current);
      }, 50);
    });
    observer.observe(node);
    return () => {
      window.clearTimeout(timer);
      observer.disconnect();
    };
  }, [points.length]);

  function autoscale() {
    const graphDiv = plotRef.current;
    if (!graphDiv) return;
    Plotly.relayout(graphDiv, {
      "xaxis.autorange": true,
      "yaxis.autorange": true,
    });
  }

  return (
    <section className="card flex h-full min-h-[640px] min-w-0 flex-col p-3 xl:min-h-0">
      <div className="mb-1 flex shrink-0 items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-wide text-slate-500">
          {t("embeddingViz", { projection: projection.toUpperCase() })}
        </h2>
        <div className="flex items-center gap-3">
          <div className="text-xs text-slate-400">{t("plotLegend")}</div>
          <button
            className="rounded-lg border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
            disabled={points.length === 0}
            onClick={autoscale}
            type="button"
          >
            {t("autoscale")}
          </button>
        </div>
      </div>
      <div className="relative min-h-0 min-w-0 flex-1" ref={frameRef}>
        {points.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">
            {t("emptyPlot")}
          </div>
        ) : (
          <>
            <div className="pointer-events-none absolute inset-0 z-[3] overflow-hidden">
              {pies.map((pie) => (
                <div
                  key={pie.id}
                  style={{
                    position: "absolute",
                    left: pie.left,
                    top: pie.top,
                    width: pie.size,
                    height: pie.size,
                    transform: "translate(-50%, -50%)",
                    borderRadius: "50%",
                    background: pieFill(pie.colors, palette[0].fill),
                    opacity: 1,
                    boxShadow: `0 0 0 ${pie.strokeWidth}px ${pie.stroke}`,
                    border: "none",
                    zIndex: pie.id === hover?.id ? 2 : 1,
                  }}
                />
              ))}
            </div>
            {hover && hoverPoint ? (
              <div
                className="pointer-events-none absolute z-[4] w-max max-w-xs rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left text-xs leading-5 text-slate-600 shadow-card"
                style={{
                  left: hover.left,
                  top: hover.top,
                  transform: hover.top < 64 ? "translate(-50%, 12px)" : "translate(-50%, calc(-100% - 10px))",
                }}
              >
                <div className="font-semibold text-slate-800">{hoverPoint.name}</div>
                {hoverPoint.kind === "disease" ? (
                  hoverPoint.avg_similarity != null ? (
                    <>
                      <div>
                        {t("hoverFrequency")}:{" "}
                        <span className="font-semibold text-slate-800">{hoverPoint.frequency || 0}</span>
                      </div>
                      <div>
                        {t("hoverAvgSimilarity")}:{" "}
                        <span className="font-semibold text-slate-800">{hoverPoint.avg_similarity.toFixed(3)}</span>
                      </div>
                    </>
                  ) : (
                    <div>{t("hoverGroundTruth")}</div>
                  )
                ) : hoverPoint.precision != null && hoverPoint.recall != null ? (
                  <>
                    <div>
                      {t("hoverPrecision")}:{" "}
                      <span className="font-semibold text-slate-800">{hoverPoint.precision.toFixed(2)}</span>
                      <span className="text-slate-400">
                        {" "}
                        ({hoverPoint.true_positives}/{hoverPoint.predicted_count})
                      </span>
                    </div>
                    <div>
                      {t("hoverRecall")}:{" "}
                      <span className="font-semibold text-slate-800">{hoverPoint.recall.toFixed(2)}</span>
                      <span className="text-slate-400">
                        {" "}
                        ({hoverPoint.true_positives}/{hoverPoint.ground_truth_count})
                      </span>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
            <Plot
              config={{
                displayModeBar: false,
                scrollZoom: true,
                doubleClick: "reset",
              }}
              data={[diseaseTrace, symptomTrace]}
              layout={{
                autosize: true,
                dragmode: "pan",
                hovermode: "closest",
                uirevision: viewRevision,
                margin: { l: 8, r: 8, t: 12, b: 8 },
                paper_bgcolor: "white",
                plot_bgcolor: "#f8fafc",
                showlegend: false,
                shapes,
                xaxis: {
                  zeroline: false,
                  showgrid: false,
                  showticklabels: false,
                  showline: false,
                  ticks: "",
                  title: "",
                  automargin: false,
                },
                yaxis: {
                  zeroline: false,
                  showgrid: false,
                  showticklabels: false,
                  showline: false,
                  ticks: "",
                  title: "",
                  automargin: false,
                },
              }}
              onClick={(event: { points?: { customdata?: unknown }[] }) => {
                const id = pointId(event.points?.[0]?.customdata);
                if (id) onSelect(id);
              }}
              onHover={(event: PlotHoverEvent) => {
                const pt = event.points?.[0];
                const id = pointId(pt?.customdata);
                if (!id || !pt) return;
                setHovered(id, hoverAnchor(pt, event.event));
              }}
              onUnhover={() => setHovered(null)}
              onInitialized={(_figure: unknown, graphDiv: PlotGraph) => syncPies(graphDiv)}
              onRelayout={() => {
                if (plotRef.current) syncPies(plotRef.current);
              }}
              onUpdate={(_figure: unknown, graphDiv: PlotGraph) => syncPies(graphDiv)}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </>
        )}
      </div>
      <div className="mt-1 flex shrink-0 flex-wrap items-center gap-3 text-xs text-slate-500">
        {symptoms.map((item) => (
          <MetricsHover key={item.id} metric={symptomMetric(item)}>
            <span className="inline-flex items-center gap-1">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{
                  background: colorMap[item.id],
                  boxShadow: `0 0 0 1.5px ${strokeMap[item.id] || MIXED_OUTLINE}`,
                }}
              />
              {item.name}
            </span>
          </MetricsHover>
        ))}
        {symptoms.length > 0 ? <span className="ml-2">{t("sizeLegend")}</span> : null}
        {symptoms.some((item) => item.id === focusedId) ? (
          <span className="ml-2">{t("linesLegend")}</span>
        ) : null}
      </div>
    </section>
  );
}
