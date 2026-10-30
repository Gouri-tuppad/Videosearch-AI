import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { FiClock, FiDatabase, FiList, FiTrendingUp } from "react-icons/fi";
import { getErrorMessage, uploadVideo, processVideo, searchVideo, getDetectedObjects, type ProcessingStatus, type SearchResponse, type UploadResponse, type DetectedObject } from "./api";
import Toast from "./components/Toast";
import UploadDropzone from "./components/UploadDropzone";
import SearchBar from "./components/SearchBar";
import StatsCard from "./components/StatsCard";
import ResultCard from "./components/ResultCard";
import ProcessingStage from "./components/ProcessingStage";
import Timeline from "./components/Timeline";

const defaultTimeline = [
  { label: "Upload video", progress: 20 },
  { label: "Run object detection", progress: 45 },
  { label: "Build embeddings", progress: 70 },
  { label: "Index search data", progress: 90 },
  { label: "Ready for search", progress: 100 },
];

type UploadPhase = "idle" | "uploading" | "success" | "ready";

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function getProgressFromStatus(status: string) {
  if (status === "running") return 65;
  if (status === "completed") return 100;
  if (status === "failed") return 0;
  return 15;
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  type HistoryItem = {
    query: string;
    timestamp?: string;
    similarity?: number;
    found: boolean;
  };
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [toast, setToast] = useState<{ message: string; kind: "success" | "error" | "info" } | null>(null);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [loadingProcess, setLoadingProcess] = useState(false);
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const searchRequestId = useRef(0);

  const timelineEvents = useMemo(
    () => defaultTimeline.map((event) => ({ ...event, progress: clamp(event.progress + getProgressFromStatus(processingStatus?.status ?? ""), 0, 100) })),
    [processingStatus],
  );

  const handleSelectFile = (file: File) => {
    searchRequestId.current += 1;
    setSelectedFile(file);
    setUploadResult(null);
    setProcessingStatus(null);
    setSearchResult(null);
    setHistory([]);
    setUploadPhase("idle");
    setDetectedObjects([]);
    setToast({ message: `Selected ${file.name}`, kind: "info" });
  };

  const handleQueryChange = (value: string) => {
    setSearchQuery(value);
    // When user edits the query, clear previous search result/preview so
    // a stale clip isn't shown while typing a new request.
    setSearchResult(null);
    setShowPreview(false);
  };

  const handleUpload = async () => {
  if (!selectedFile) {
    setToast({
      message: "Choose a video before uploading.",
      kind: "error",
    });
    return;
  }

  setLoadingUpload(true);
  searchRequestId.current += 1;
  setUploadPhase("uploading");
  setToast(null);

  try {
    const response = await uploadVideo(selectedFile);

    console.log("Upload Response:", response);

    setUploadResult(response);
    setProcessingStatus(null);
    setSearchResult(null);
    setHistory([]);
    setDetectedObjects([]);

    setUploadPhase("ready");

    setToast({
      message: `Upload Successful: ${response.filename}`,
      kind: "success",
    });
  } catch (error: any) {
    console.error("Upload Error:", error);
    console.error("Response:", error?.response);
    console.error("Data:", error?.response?.data);

    setUploadResult(null);
    setUploadPhase("idle");

    setToast({
      message: getErrorMessage(error),
      kind: "error",
    });
  } finally {
    setLoadingUpload(false);
  }
};
  const handleProcess = async () => {
    if (!uploadResult?.filename) {
      setToast({ message: "Upload a video before processing.", kind: "error" });
      return;
    }
    setLoadingProcess(true);
    try {
      const status = await processVideo(uploadResult.filename);
      setProcessingStatus(status);
      // Fetch real detected objects from DB after processing
      try {
        const objectsResponse = await getDetectedObjects(uploadResult.filename);
        setDetectedObjects(objectsResponse.detected_objects);
      } catch (objError) {
        console.warn("Could not fetch detected objects:", objError);
        // Fallback to detected_objects from processing response if available
        if (status.detected_objects) {
          setDetectedObjects(status.detected_objects.map((name: string) => ({
            object_name: name,
            first_timestamp: 0,
            detection_count: 0,
            max_confidence: 0,
          })));
        }
      }
      setToast({ message: "Video processed successfully.", kind: "success" });
    } catch (error) {
      console.error("Processing error:", error);
      setToast({ message: "Processing failed. See backend logs.", kind: "error" });
    } finally {
      setLoadingProcess(false);
    }
  };

  const handleSearch = async (overrideQuery?: string) => {
    const rawQuery = typeof overrideQuery === "string" ? overrideQuery : searchQuery;
    const normalizedQuery = rawQuery.trim();
    if (!normalizedQuery) {
      setToast({ message: "Enter a search query first.", kind: "error" });
      return;
    }
    if (typeof overrideQuery === "string") {
      setSearchQuery(overrideQuery);
    }
    const requestId = ++searchRequestId.current;
    setSearchResult(null);
    setShowPreview(false);
    setLoadingSearch(true);
    try {
      const response = await searchVideo(normalizedQuery, uploadResult?.filename);
      if (requestId !== searchRequestId.current) return;
      // always set the latest response
      setSearchResult(response);
      const isFound = response.status === "found" || Boolean(response.found);
      if (isFound) {
        setShowPreview(true);
        setToast({ message: "Match found successfully!", kind: "success" });
      } else {
        setShowPreview(false);
        setToast({
          message: response.message || `Object "${normalizedQuery}" not found in this video.`,
          kind: "info",
        });
      }

      setHistory((prev) => [
        {
          query: normalizedQuery,
          timestamp: response.timestamp,
          similarity: response.similarity_score ?? response.results?.[0]?.score,
          found: isFound,
        },
        ...prev,
      ].slice(0, 6));
    } catch (error: any) {
      console.error("Search error:", error);
      setToast({ message: getErrorMessage(error), kind: "error" });
      // clear result on failure to avoid reusing previous result
      setSearchResult(null);
      setShowPreview(false);
    } finally {
      setLoadingSearch(false);
    }
  };

  const confidenceMeter = useMemo(() => {
    if (searchResult?.similarity_score !== undefined && searchResult?.similarity_score !== null) {
      return clamp(searchResult.similarity_score * 100, 0, 100);
    }
    if (searchResult?.results && searchResult.results.length > 0) {
      return clamp(searchResult.results[0].score * 100, 0, 100);
    }
    return 0;
  }, [searchResult]);

  const uploadStatusText = loadingUpload
    ? "Uploading"
    : uploadPhase === "success"
      ? "Upload Successful"
      : uploadPhase === "ready"
        ? "Ready to Process"
        : selectedFile
          ? "Ready to Upload"
          : "Idle";

  return (
    <div className="relative overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.16),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(168,85,247,0.16),_transparent_26%)] px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto grid max-w-7xl gap-8">
        <header className="glass-card border-white/10 bg-slate-950/80 p-6 shadow-glass">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-sky-300/90">AI Video Search</p>
              <h1 className="mt-4 text-3xl font-semibold text-white sm:text-4xl">Natural Language Video Search Dashboard</h1>
              <p className="mt-3 max-w-2xl text-slate-400">Upload a video, process it with object detection and embeddings, then ask questions in natural language to retrieve the most relevant clip instantly.</p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-slate-900/90 px-5 py-4">
                <p className="text-sm text-slate-400">Current storage</p>
                <p className="mt-2 text-2xl font-semibold text-white">{uploadResult?.filename ?? "No video uploaded"}</p>
              </div>
              <div className="rounded-3xl border border-white/10 bg-slate-900/90 px-5 py-4">
                <p className="text-sm text-slate-400">Processing state</p>
                <p className="mt-2 text-2xl font-semibold text-white">{processingStatus?.status ?? "idle"}</p>
              </div>
            </div>
          </div>
        </header>

        <main className="grid gap-8 xl:grid-cols-[1.35fr_0.85fr]">
          <section className="space-y-6">
            <UploadDropzone
              isDragging={isDragging}
              selectedFileName={selectedFile?.name ?? null}
              onSelectFile={handleSelectFile}
              onDragStateChange={setIsDragging}
            />

            <div className="glass-card rounded-3xl border border-white/10 bg-slate-950/60 px-5 py-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Upload state</p>
                  <p className="mt-1 text-lg font-semibold text-white">{uploadStatusText}</p>
                </div>
                {uploadResult?.filename ? (
                  <p className="text-sm text-emerald-300">Uploaded file: {uploadResult.filename}</p>
                ) : selectedFile ? (
                  <p className="text-sm text-slate-400">Selected file: {selectedFile.name}</p>
                ) : null}
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <button
                type="button"
                disabled={!selectedFile || loadingUpload}
                onClick={handleUpload}
                className="glass-card flex items-center justify-between gap-3 rounded-3xl px-6 py-5 text-sm font-semibold text-white transition hover:border-sky-400/50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span>Upload selected video</span>
                <span>{loadingUpload ? "Uploading…" : uploadResult ? "Upload successful" : "Upload"}</span>
              </button>
              <button
                type="button"
                disabled={!uploadResult || loadingProcess || uploadPhase !== "ready"}
                onClick={handleProcess}
                className="glass-card flex items-center justify-between gap-3 rounded-3xl px-6 py-5 text-sm font-semibold text-white transition hover:border-emerald-400/50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span>Process video data</span>
                <span>{loadingProcess ? "Processing…" : "Run pipeline"}</span>
              </button>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
              <div className="glass-card p-5">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-400">AI confidence gauge</p>
                    <h2 className="mt-2 text-2xl font-semibold text-white">Similarity strength</h2>
                  </div>
                  <div className="rounded-full border border-white/10 bg-slate-900/80 px-4 py-2 text-sm text-slate-200">{confidenceMeter.toFixed(0)}%</div>
                </div>
                <div className="mt-6 h-4 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-gradient-to-r from-sky-400 via-cyan-400 to-emerald-300 transition-all" style={{ width: `${confidenceMeter}%` }} />
                </div>
                <p className="mt-4 text-sm leading-6 text-slate-400">The confidence gauge updates after each search and represents the best relevance score returned by the AI search engine.</p>
              </div>

              <div className="glass-card p-5">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Search history</p>
                    <h2 className="mt-2 text-2xl font-semibold text-white">Recent queries</h2>
                  </div>
                  <FiList className="h-6 w-6 text-sky-300" />
                </div>
                <div className="mt-6 space-y-3">
                  {history.length > 0 ? (
                    history.map((item, index) => (
                      <div
                        key={index}
                        className="rounded-3xl border border-white/10 bg-slate-900/80 px-4 py-4"
                      >
                        <p className="font-semibold text-white">🔍 {item.query}</p>

                        <p className="text-sm text-slate-400">
                          {item.found ? "Match Found" : "No Match"}
                        </p>

                        {item.timestamp && (
                          <p className="text-sm text-sky-300">
                            Timestamp: {item.timestamp}
                          </p>
                        )}

                        {item.similarity !== undefined && (
                          <p className="text-sm text-emerald-300">
                            Confidence: {(item.similarity * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                    ))
                  ) : null}
                </div>
              </div>
            </div>

            <SearchBar value={searchQuery} onChange={handleQueryChange} onSearch={handleSearch} disabled={loadingSearch || !uploadResult?.filename} />

            <div className="space-y-4">
              <AnimatePresence>
                {loadingSearch ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="glass-card p-6"
                  >
                    <div className="space-y-3">
                      <div className="h-4 w-1/3 rounded-full skeleton" />
                      <div className="h-48 rounded-[28px] skeleton" />
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
              {searchResult ? (
              <>
                <ResultCard
                  status={searchResult.status}
                  query={searchResult.query}
                  title={searchResult.status === "found" || searchResult.found ? `Found clips for "${searchResult.query}"` : "No relevant clip found"}
                  timestamp={searchResult.timestamp}
                  similarity={searchResult.similarity_score}
                  message={searchResult.message}
                  suggestions={searchResult.suggestions}
                  results={searchResult.results}
                  clipPath={showPreview ? searchResult.clip_path : undefined}
                  onSuggestionClick={(suggestion) => handleSearch(suggestion)}
                />

                <div className="mt-6 flex justify-center gap-4">
                  <button
                    onClick={() => {
                      setShowPreview(false);
                      setSearchResult(null);
                      setSearchQuery("");
                      setLoadingSearch(false);
                    }}
                    className="rounded-xl bg-blue-600 px-6 py-3 text-white"
                  >
                    🔍 Search Another Object
                  </button>

                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      setUploadResult(null);
                      setProcessingStatus(null);
                      setSearchResult(null);
                      setSearchQuery("");
                      setHistory([]);
                      setUploadPhase("idle");
                      setDetectedObjects([]);
                    }}
                    className="rounded-xl bg-red-600 px-6 py-3 font-semibold text-white hover:bg-red-700"
                  >
                    📁 Upload New Video
                  </button>
                </div>
              </>
            ) : null}
            </div>
          </section>

          <aside className="space-y-6">
            <div className="glass-card p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-400">AI dashboard stats</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Operational metrics</h2>
                </div>
                <FiTrendingUp className="h-6 w-6 text-emerald-300" />
              </div>
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <StatsCard label="Uploaded video" value={uploadResult?.filename ?? "—"} />
                <StatsCard label="Pipeline status" value={processingStatus?.status ?? "idle"} accent="bg-fuchsia-500/15 text-fuchsia-200" />
                <StatsCard label="Last query" value={history[0]?.query ?? "—"} accent="bg-sky-500/15 text-sky-200" />
                <StatsCard label="Clip confidence" value={`${confidenceMeter.toFixed(0)}%`} accent="bg-emerald-500/15 text-emerald-200" />
              </div>
            </div>

            <div className="glass-card p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Processing timeline</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Pipeline progress</h2>
                </div>
                <FiClock className="h-6 w-6 text-slate-300" />
              </div>
              <div className="mt-6 space-y-4">
                {processingStatus?.steps?.slice(0, 4).map((step) => (
                  <ProcessingStage key={step.name} step={step.name} status={step.status} message={step.message} />
                ))}
              </div>
            </div>

            <Timeline events={timelineEvents} />

            <div className="glass-card p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Object detection</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Detected Objects</h2>
                </div>
                <FiDatabase className="h-6 w-6 text-fuchsia-300" />
              </div>
              <div className="mt-6 space-y-3">
                {detectedObjects.length > 0 ? (
                  detectedObjects.map((obj, idx) => (
                    <div
                      key={`${obj.object_name}-${idx}`}
                      className="flex items-center justify-between rounded-3xl border border-white/10 bg-slate-900/80 px-4 py-3"
                    >
                      <div className="min-w-0">
                        <p className="font-semibold capitalize text-white">{obj.object_name}</p>
                        <p className="text-xs text-slate-400">
                          First seen: {obj.first_timestamp > 0 ? `${obj.first_timestamp.toFixed(1)}s` : "—"}
                          {obj.detection_count > 0 && ` · ${obj.detection_count} detections`}
                        </p>
                      </div>
                      {obj.max_confidence > 0 && (
                        <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-300">
                          {(obj.max_confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-500">
                    {processingStatus?.status === "completed"
                      ? "No objects detected in this video."
                      : "Process a video to see detected objects."}
                  </p>
                )}
              </div>
            </div>
          </aside>
        </main>
      </div>

      <Toast message={toast?.message ?? ""} kind={toast?.kind ?? "info"} visible={Boolean(toast)} />
    </div>
  );
}
