export const FREQUENCY_COLORS: Record<number, string> = {
  1: "#9CA3AF",
  2: "#EAB308",
  3: "#F97316",
  4: "#EF4444",
  5: "#7F1D1D",
};

export type SymptomStyle = {
  fill: string;
  stroke: string;
};

export type ColorPalette = {
  id: string;
  name: string;
  colors: SymptomStyle[];
};

export const COLOR_PALETTES: ColorPalette[] = [
  {
    id: "vivid",
    name: "Vivid",
    colors: [
      { fill: "#2563EB", stroke: "#1E3A8A" },
      { fill: "#16A34A", stroke: "#14532D" },
      { fill: "#E11D48", stroke: "#9F1239" },
      { fill: "#D97706", stroke: "#92400E" },
      { fill: "#7C3AED", stroke: "#5B21B6" },
    ],
  },
  {
    id: "colorblind",
    name: "Colorblind-safe",
    colors: [
      { fill: "#0072B2", stroke: "#084C75" },
      { fill: "#E69F00", stroke: "#8A5A00" },
      { fill: "#009E73", stroke: "#06664A" },
      { fill: "#CC79A7", stroke: "#861B54" },
      { fill: "#D55E00", stroke: "#8A3C00" },
    ],
  },
  {
    id: "tableau",
    name: "Tableau",
    colors: [
      { fill: "#4E79A7", stroke: "#2F4B68" },
      { fill: "#F28E2B", stroke: "#A85A12" },
      { fill: "#59A14F", stroke: "#2F6A2A" },
      { fill: "#E15759", stroke: "#9B2C2E" },
      { fill: "#B07AA1", stroke: "#6E4A64" },
    ],
  },
  {
    id: "ocean",
    name: "Ocean",
    colors: [
      { fill: "#0EA5E9", stroke: "#075985" },
      { fill: "#0F766E", stroke: "#134E4A" },
      { fill: "#6366F1", stroke: "#312E81" },
      { fill: "#F59E0B", stroke: "#92400E" },
      { fill: "#F43F5E", stroke: "#9F1239" },
    ],
  },
  {
    id: "jewel",
    name: "Jewel",
    colors: [
      { fill: "#0F766E", stroke: "#134E4A" },
      { fill: "#C026D3", stroke: "#86198F" },
      { fill: "#1D4ED8", stroke: "#1E3A8A" },
      { fill: "#CA8A04", stroke: "#854D0E" },
      { fill: "#BE123C", stroke: "#881337" },
    ],
  },
  {
    id: "high-contrast",
    name: "High contrast",
    colors: [
      { fill: "#1D4ED8", stroke: "#172554" },
      { fill: "#B91C1C", stroke: "#7F1D1D" },
      { fill: "#15803D", stroke: "#14532D" },
      { fill: "#A21CAF", stroke: "#701A75" },
      { fill: "#EA580C", stroke: "#7C2D12" },
    ],
  },
];

export const DEFAULT_PALETTE_ID = COLOR_PALETTES[0].id;

export const SYMPTOM_PALETTE: SymptomStyle[] = COLOR_PALETTES[0].colors;

export const MIXED_OUTLINE = "#111827";

export function getPalette(id?: string | null): ColorPalette {
  return COLOR_PALETTES.find((item) => item.id === id) || COLOR_PALETTES[0];
}

export function frequencyColor(frequency: number): string {
  return FREQUENCY_COLORS[Math.min(5, Math.max(1, frequency))] || FREQUENCY_COLORS[1];
}

export function frequencyLabel(frequency: number, selectedCount: number): string {
  return `${frequency}/${selectedCount || frequency}`;
}

export function symptomStyle(index: number, palette: SymptomStyle[] = SYMPTOM_PALETTE): SymptomStyle {
  return palette[index % palette.length];
}

export function symptomColor(index: number, palette: SymptomStyle[] = SYMPTOM_PALETTE): string {
  return symptomStyle(index, palette).fill;
}

export function symptomColorMap(ids: string[], palette: SymptomStyle[] = SYMPTOM_PALETTE): Record<string, string> {
  return Object.fromEntries(ids.map((id, index) => [id, symptomStyle(index, palette).fill]));
}

export function symptomStrokeMap(ids: string[], palette: SymptomStyle[] = SYMPTOM_PALETTE): Record<string, string> {
  return Object.fromEntries(ids.map((id, index) => [id, symptomStyle(index, palette).stroke]));
}

export function pieFill(colors: string[], fallback = SYMPTOM_PALETTE[0].fill): string {
  if (colors.length === 0) return fallback;
  if (colors.length === 1) return colors[0];
  const slice = 100 / colors.length;
  const stops = colors.map((color, index) => {
    const start = index * slice;
    const end = (index + 1) * slice;
    return `${color} ${start}% ${end}%`;
  });
  return `conic-gradient(${stops.join(", ")})`;
}

export function pieStroke(fills: string[], palette: SymptomStyle[] = SYMPTOM_PALETTE): string {
  if (fills.length === 1) {
    return palette.find((item) => item.fill === fills[0])?.stroke || MIXED_OUTLINE;
  }
  return MIXED_OUTLINE;
}
