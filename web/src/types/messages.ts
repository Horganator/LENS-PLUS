export type ObjectDetection = {
  label: string;
  confidence: number;
  bbox?: [number, number, number, number];
};

export type DirectionalContext = {
  sample_timestamp_ms?: number | null;
  server_received_at?: string;
  age_ms?: number;
  is_stale?: boolean;
  rotation_rate_dps?: {
    alpha?: number | null;
    beta?: number | null;
    gamma?: number | null;
  } | null;
  orientation_deg?: {
    alpha?: number | null;
    beta?: number | null;
    gamma?: number | null;
    absolute?: boolean | null;
  } | null;
};

export type InferenceMessage = {
  timestamp: string;
  guidance_text?: string;
  scene_summary?: string;
  directional?: DirectionalContext | null;
  objects?: ObjectDetection[];
};

export function isInferenceMessage(value: unknown): value is InferenceMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<InferenceMessage>;
  return typeof candidate.timestamp === "string";
}
