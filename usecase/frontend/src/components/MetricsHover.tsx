import type { ReactNode } from "react";
import type { MessageKey } from "../i18n";
import { useI18n } from "../i18n";
import type { SymptomMetric } from "../types";

type Translate = (key: MessageKey, vars?: Record<string, string | number>) => string;

export function metricById(metrics: SymptomMetric[] | undefined, id: string): SymptomMetric | undefined {
  return metrics?.find((item) => item.id === id);
}

export function metricsLabel(metric: SymptomMetric, t: Translate): string {
  return [
    `${t("hoverPrecision")}: ${metric.precision.toFixed(2)} (${metric.true_positives}/${metric.predicted_count})`,
    `${t("hoverRecall")}: ${metric.recall.toFixed(2)} (${metric.true_positives}/${metric.ground_truth_count})`,
    t("metricsCutoff", { k: metric.top_k }),
  ].join("\n");
}

type Props = {
  metric?: SymptomMetric | null;
  children: ReactNode;
  className?: string;
};

export function MetricsHover({ metric, children, className = "" }: Props) {
  const { t } = useI18n();
  if (!metric) return <span className={className}>{children}</span>;
  return (
    <span className={`group/metric relative inline-flex ${className}`} title={metricsLabel(metric, t)}>
      {children}
      <span className="pointer-events-none absolute left-1/2 top-full z-40 mt-1 hidden w-max max-w-xs -translate-x-1/2 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left text-xs leading-5 text-slate-600 shadow-card group-hover/metric:block">
        <div>
          {t("hoverPrecision")}: <span className="font-semibold text-slate-800">{metric.precision.toFixed(2)}</span>
          <span className="text-slate-400"> ({metric.true_positives}/{metric.predicted_count})</span>
        </div>
        <div>
          {t("hoverRecall")}: <span className="font-semibold text-slate-800">{metric.recall.toFixed(2)}</span>
          <span className="text-slate-400"> ({metric.true_positives}/{metric.ground_truth_count})</span>
        </div>
        <div className="text-[10px] text-slate-400">{t("metricsCutoff", { k: metric.top_k })}</div>
      </span>
    </span>
  );
}
