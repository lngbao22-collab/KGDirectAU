import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Lang = "en" | "vi";

const STORAGE_KEY = "kgau-lang";

const MESSAGES = {
  en: {
    title: "KGAU Biomedical Disease Prediction",
    subtitle:
      "Demonstration of KGAU-trained embeddings for disease prediction from symptoms on Hetionet. Not a medical diagnosis tool.",
    working: "Working…",
    predictionFailed: "Prediction failed",
    loadModelFailed: "Could not load model",
    loadDiseaseFailed: "Could not load disease",
    language: "Language",
    english: "English",
    vietnamese: "Tiếng Việt",
    columns: "Columns",
    input: "Input",
    plot: "Visualization",
    info: "Details",
    hide: "Hide",
    keepOneColumn: "Keep at least one column visible",
    hideColumn: "Hide {label}",
    showColumn: "Show {label}",
    resizeColumns: "Resize {left} and {right}",
    advancedSettings: "Advanced Settings",
    backboneModel: "Backbone Model",
    unavailable: "(unavailable)",
    defaultSuffix: "(default)",
    embeddingProjection: "Embedding Projection",
    umap: "UMAP (2D)",
    tsne: "t-SNE (2D)",
    symptomColorCombo: "Symptom Color Combo",
    palette_vivid: "Vivid",
    palette_colorblind: "Colorblind-safe",
    palette_tableau: "Tableau",
    palette_ocean: "Ocean",
    palette_jewel: "Jewel",
    palette_highContrast: "High contrast",
    inputSymptoms: "Input Symptoms",
    maxSymptoms: "Maximum of 5 symptoms",
    addSymptom: "Add symptom...",
    predicting: "Predicting...",
    predict: "Predict",
    clear: "Clear",
    focusedEntity: "Focused Entity",
    emptyDetail: "Click a candidate disease or an input symptom to inspect metadata.",
    symptom: "Symptom",
    noWiki: "No Wikipedia page in the current database.",
    matchedSymptoms: "Matched Symptoms ({label})",
    similarDiseases: "Similar Diseases (Resembles)",
    graphLookup: "Graph lookup · no model inference",
    noResembles: "No Resembles neighbors in this subset.",
    candidateDiseases: "Candidate Diseases (Top {n})",
    search: "Search",
    download: "Download",
    sortFrequency: "Sort by Frequency (desc)",
    sortAvg: "Sort by Avg Similarity",
    sortMax: "Sort by Max Similarity",
    sortName: "Sort by Name",
    rank: "Rank",
    disease: "Disease",
    matchedFreq: "Matched Symptoms (Frequency)",
    matchedNames: "Matched Symptoms (Names)",
    avgSimilarity: "Avg Similarity",
    maxSimilarity: "Max Similarity",
    hoverPrecision: "Precision",
    hoverRecall: "Recall",
    metricsCutoff: "Top-{k} vs ground-truth Presents",
    page: "Page {page} / {pageCount}",
    prev: "Prev",
    next: "Next",
    embeddingViz: "Embedding Visualization ({projection})",
    plotLegend: "▲ symptoms · ● diseases",
    autoscale: "Autoscale",
    emptyPlot: "Select symptoms and click Predict to project the embedding space.",
    sizeLegend: "Size = average similarity",
    linesLegend: "Lines = ground-truth diseases",
    hoverFrequency: "Frequency",
    hoverAvgSimilarity: "Avg similarity",
    hoverGroundTruth: "Ground-truth disease",
    traceDiseases: "Diseases",
    traceSymptoms: "Input Symptoms",
  },
  vi: {
    title: "Dự đoán bệnh y sinh KGAU",
    subtitle:
      "Minh họa embedding huấn luyện KGAU để dự đoán bệnh từ triệu chứng trên Hetionet. Không phải công cụ chẩn đoán y khoa.",
    working: "Đang xử lý…",
    predictionFailed: "Dự đoán thất bại",
    loadModelFailed: "Không thể tải mô hình",
    loadDiseaseFailed: "Không thể tải thông tin bệnh",
    language: "Ngôn ngữ",
    english: "English",
    vietnamese: "Tiếng Việt",
    columns: "Cột",
    input: "Đầu vào",
    plot: "Trực quan",
    info: "Chi tiết",
    hide: "Ẩn",
    keepOneColumn: "Giữ ít nhất một cột hiển thị",
    hideColumn: "Ẩn {label}",
    showColumn: "Hiện {label}",
    resizeColumns: "Đổi kích thước {left} và {right}",
    advancedSettings: "Cài đặt nâng cao",
    backboneModel: "Mô hình xương sống",
    unavailable: "(không khả dụng)",
    defaultSuffix: "(mặc định)",
    embeddingProjection: "Phép chiếu embedding",
    umap: "UMAP (2D)",
    tsne: "t-SNE (2D)",
    symptomColorCombo: "Tổ hợp màu triệu chứng",
    palette_vivid: "Rực rỡ",
    palette_colorblind: "Bảng màu cho người mù màu",
    palette_tableau: "Tableau",
    palette_ocean: "Đại dương",
    palette_jewel: "Ngọc",
    palette_highContrast: "Tương phản cao",
    inputSymptoms: "Triệu chứng đầu vào",
    maxSymptoms: "Tối đa 5 triệu chứng",
    addSymptom: "Thêm triệu chứng...",
    predicting: "Đang dự đoán...",
    predict: "Dự đoán",
    clear: "Xóa",
    focusedEntity: "Thực thể đang xem",
    emptyDetail: "Chọn 1 bệnh hoặc triệu chứng để xem thông tin.",
    symptom: "Triệu chứng",
    noWiki: "Không có trang Wikipedia trong cơ sở dữ liệu hiện tại.",
    matchedSymptoms: "Triệu chứng khớp ({label})",
    similarDiseases: "Bệnh tương tự (Resembles)",
    graphLookup: "Tra cứu đồ thị · không suy luận mô hình",
    noResembles: "Không có đỉnh lân cận Resembles trong tập con này.",
    candidateDiseases: "Các bệnh dự đoán (Top {n})",
    search: "Tìm kiếm",
    download: "Tải xuống",
    sortFrequency: "Sắp xếp theo tần suất (giảm dần)",
    sortAvg: "Sắp xếp theo độ tương đồng TB",
    sortMax: "Sắp xếp theo độ tương đồng max",
    sortName: "Sắp xếp theo tên",
    rank: "Hạng",
    disease: "Bệnh",
    matchedFreq: "Triệu chứng khớp (tần suất)",
    matchedNames: "Triệu chứng khớp (tên)",
    avgSimilarity: "Độ tương đồng TB",
    maxSimilarity: "Độ tương đồng max",
    hoverPrecision: "Độ chính xác",
    hoverRecall: "Độ phủ",
    metricsCutoff: "Top-{k} so với cạnh Presents ground-truth",
    page: "Trang {page} / {pageCount}",
    prev: "Trước",
    next: "Sau",
    embeddingViz: "Trực quan embedding ({projection})",
    plotLegend: "▲ triệu chứng · ● bệnh",
    autoscale: "Tự căn tỷ lệ",
    emptyPlot: "Chọn triệu chứng rồi nhấn Dự đoán để chiếu không gian embedding.",
    sizeLegend: "Kích thước = độ tương đồng trung bình",
    linesLegend: "Đường = bệnh ground-truth",
    hoverFrequency: "Tần suất",
    hoverAvgSimilarity: "Độ tương đồng TB",
    hoverGroundTruth: "Bệnh ground-truth",
    traceDiseases: "Bệnh",
    traceSymptoms: "Triệu chứng đầu vào",
  },
} as const;

export type MessageKey = keyof typeof MESSAGES.en;

function readStoredLang(): Lang {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === "vi" ? "vi" : "en";
  } catch {
    return "en";
  }
}

function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, key: string) => String(vars[key] ?? `{${key}}`));
}

type I18nValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(readStoredLang);

  useEffect(() => {
    document.documentElement.lang = lang;
    document.title = MESSAGES[lang].title;
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* ignore quota / private mode */
    }
  }, [lang]);

  const value = useMemo<I18nValue>(
    () => ({
      lang,
      setLang: setLangState,
      t: (key, vars) => interpolate(MESSAGES[lang][key] || MESSAGES.en[key], vars),
    }),
    [lang],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const value = useContext(I18nContext);
  if (!value) throw new Error("useI18n must be used within I18nProvider");
  return value;
}

export function LanguageToggle() {
  const { lang, setLang, t } = useI18n();
  return (
    <div className="flex items-center gap-2" title={t("language")}>
      <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{t("language")}</span>
      <div className="flex rounded-lg border border-slate-200 p-0.5">
        {(["en", "vi"] as const).map((item) => (
          <button
            className={`rounded-md px-2 py-1 text-xs font-medium ${
              lang === item ? "bg-brand-600 text-white" : "text-slate-500 hover:bg-slate-50"
            }`}
            key={item}
            onClick={() => setLang(item)}
            type="button"
          >
            {item === "en" ? "EN" : "VI"}
          </button>
        ))}
      </div>
    </div>
  );
}
