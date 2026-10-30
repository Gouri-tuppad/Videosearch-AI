import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use(
  (config) => {
    console.log("[Axios request]", config.method?.toUpperCase(), config.url);
    console.log("[Axios request payload]", config.data);
    return config;
  },
  (error) => {
    console.error("[Axios request error]", error);
    return Promise.reject(error);
  },
);

api.interceptors.response.use(
  (response) => {
    console.log("[Axios response]", response.status, response.config.url);
    console.log("[Axios response data]", response.data);
    return response;
  },
  (error) => {
    console.error("[Axios error object]", error);
    console.error("[Axios error response]", error.response);
    console.error("[Axios error message]", error.message);
    console.error("[Axios error code]", error.code);
    return Promise.reject(error);
  },
);

export type UploadResponse = {
  filename: string;
  size: number;
  upload_status: string;
  processing_status: string;
};

export type SearchResultItem = {
  label: string;
  start: number;
  end: number;
  clip_url: string;
  score: number;
  start_time?: number;
  end_time?: number;
  best_similarity_score?: number;
  clip_path?: string;
  representative_frame?: string;
  video_name?: string;
};

export type SearchResponse = {
  status: "found" | "not_found";
  query: string;
  results: SearchResultItem[];
  message?: string;
  suggestions?: string[];
  found?: boolean;
  video_name?: string;
  threshold?: number;
  timestamp?: string;
  similarity_score?: number;
  frame_name?: string;
  clip_path?: string;
  best_similarity_score?: number;
  error?: string;
};

export type ProcessingStatus = {
  video_name?: string;
  status: string;
  current_step?: string;
  error?: string | null;
  detected_objects?: string[];
  steps?: Array<{ name: string; status: string; message?: string | null }>;
};

export function getErrorMessage(error: unknown): string {
  const maybeError = error as {
    response?: { data?: unknown; status?: number };
    message?: string;
    code?: string;
  };

  const responseData = maybeError.response?.data;
  if (typeof responseData === "string") {
    return responseData;
  }

  if (typeof responseData === "object" && responseData && "detail" in responseData) {
    const detail = (responseData as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
  }

  if (typeof responseData === "object" && responseData && "message" in responseData) {
    const message = (responseData as { message?: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }

  if (maybeError.message && maybeError.message !== "Network Error") {
    return maybeError.message;
  }

  if (maybeError.code === "ERR_NETWORK") {
    return "Network error while uploading the video.";
  }

  return "Upload failed.";
}

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await api.post<UploadResponse>("/videos/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    console.log("[Upload response]", response);
    console.log("[Upload response data]", response.data);
    return response.data;
  } catch (error) {
    console.error("[Upload error object]", error);
    console.error("[Upload error response]", (error as { response?: unknown }).response);
    console.error("[Upload error message]", (error as { message?: string }).message);
    console.error("[Upload error code]", (error as { code?: string }).code);
    throw error;
  }
}

export async function processVideo(filename: string): Promise<ProcessingStatus> {
  const response = await api.post<ProcessingStatus>(`/detections/process/${encodeURIComponent(filename)}`);
  return response.data;
}

export async function searchVideo(query: string, videoName?: string): Promise<SearchResponse> {
  const payload = { query, video_name: videoName };
  const response = await api.post<SearchResponse>("/search", payload);
  return response.data;
}

export type DetectedObject = {
  object_name: string;
  first_timestamp: number;
  detection_count: number;
  max_confidence: number;
};

export type DetectedObjectsResponse = {
  video_name: string;
  video_id: number;
  detected_objects: DetectedObject[];
  total_unique_objects: number;
};

export async function getDetectedObjects(filename: string): Promise<DetectedObjectsResponse> {
  const response = await api.get<DetectedObjectsResponse>(`/detections/objects/${encodeURIComponent(filename)}`);
  return response.data;
}

export async function healthCheck() {
  return api.get("/health");
}
