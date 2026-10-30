import { motion } from "framer-motion";
import { FiAlertCircle, FiCheckCircle, FiDownload, FiPlay, FiSearch, FiTag } from "react-icons/fi";
import type { SearchResultItem } from "../api";

interface ResultCardProps {
  status?: "found" | "not_found";
  title?: string;
  query?: string;
  timestamp?: string;
  similarity?: number;
  message?: string;
  clipPath?: string;
  suggestions?: string[];
  results?: SearchResultItem[];
  onSuggestionClick?: (suggestion: string) => void;
}

const API_URL = "http://127.0.0.1:8000";

function formatSeconds(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function resolveMediaUrl(path: string): string {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_URL}/${path.replace(/^\/+/, "")}`;
}

export default function ResultCard({
  status,
  title,
  query,
  timestamp,
  similarity,
  message,
  clipPath,
  suggestions,
  results,
  onSuggestionClick,
}: ResultCardProps) {
  const isNotFound = status === "not_found";
  const hasMultipleResults = results && results.length > 0;
  const singleVideoUrl = resolveMediaUrl(clipPath || "");

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="glass-card overflow-hidden"
    >
      {isNotFound ? (
        <div className="p-6">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-2xl bg-amber-500/15 text-amber-400">
              <FiAlertCircle className="h-6 w-6" />
            </div>
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-[0.24em] text-amber-400">Search Result</p>
              <h3 className="text-xl font-semibold text-white">
                {query ? `No match for "${query}"` : "Not found in this video"}
              </h3>
              <p className="text-sm leading-relaxed text-slate-300">
                {message || "Not found in this video. Try another search."}
              </p>
            </div>
          </div>

          {suggestions && suggestions.length > 0 && (
            <div className="mt-6 border-t border-white/10 pt-5">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Suggestions from this video:</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => onSuggestionClick?.(suggestion)}
                    className="inline-flex items-center gap-1.5 rounded-full border border-sky-400/30 bg-sky-500/15 px-3.5 py-1.5 text-xs font-semibold text-sky-200 transition hover:border-sky-400/60 hover:bg-sky-500/25 active:scale-95"
                  >
                    <FiSearch className="h-3 w-3" />
                    <span>{suggestion}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <FiCheckCircle className="h-4 w-4 text-emerald-400" />
                <p className="text-xs uppercase tracking-[0.24em] text-emerald-400">Result Found</p>
              </div>

              <h3 className="mt-1 text-xl font-semibold text-white">
                {title || (query ? `Clips for "${query}"` : "Top-ranked AI clip")}
              </h3>

              {message && (
                <p className="mt-1 text-sm text-slate-400">
                  {message}
                </p>
              )}
            </div>

            <div className="flex flex-wrap gap-3">
              {timestamp && !hasMultipleResults && (
                <span className="rounded-full bg-slate-900/90 px-4 py-2 text-sm text-slate-200">
                  Timestamp: {timestamp}
                </span>
              )}

              {similarity !== undefined && !hasMultipleResults && (
                <span className="rounded-full bg-slate-900/90 px-4 py-2 text-sm text-slate-200">
                  Confidence: {(similarity * 100).toFixed(1)}%
                </span>
              )}

              {hasMultipleResults && (
                <span className="rounded-full bg-slate-900/90 px-4 py-2 text-sm text-slate-200">
                  {results.length} {results.length === 1 ? "clip found" : "clips found"}
                </span>
              )}
            </div>
          </div>

          {/* Multiple clips list */}
          {hasMultipleResults ? (
            <div className="divide-y divide-white/10 border-t border-white/10">
              {results.map((item, idx) => {
                const itemVideoUrl = resolveMediaUrl(item.clip_url || item.clip_path || "");
                const timeStr = `${formatSeconds(item.start)} - ${formatSeconds(item.end)} (${item.start.toFixed(1)}s - ${item.end.toFixed(1)}s)`;

                return (
                  <div key={`${item.clip_url}-${idx}`} className="bg-slate-950/60 p-5 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-sky-300">
                          <FiTag className="h-3 w-3" />
                          {item.label}
                        </span>
                        <span className="text-sm font-medium text-slate-300">
                          ⏱ {timeStr}
                        </span>
                      </div>
                      {item.score !== undefined && (
                        <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-semibold text-emerald-300">
                          Confidence: {(item.score * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>

                    {itemVideoUrl && (
                      <video
                        controls
                        preload="metadata"
                        className="w-full rounded-2xl bg-black shadow-lg"
                        key={itemVideoUrl}
                      >
                        <source src={itemVideoUrl} type="video/mp4" />
                      </video>
                    )}

                    <div className="flex flex-wrap gap-3">
                      <a
                        href={itemVideoUrl}
                        download
                        className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-5 py-2.5 text-xs font-semibold text-slate-950 hover:bg-sky-400 transition"
                      >
                        <FiDownload />
                        Download Clip
                      </a>
                      <button
                        type="button"
                        onClick={() => window.open(itemVideoUrl, "_blank")}
                        className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/80 px-5 py-2.5 text-xs font-semibold text-slate-100 hover:bg-slate-800 transition"
                      >
                        <FiPlay />
                        Preview in New Tab
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : singleVideoUrl ? (
            /* Single clip fallback */
            <div className="border-t border-white/10 bg-slate-950/60 px-5 py-4">
              <video
                controls
                preload="metadata"
                className="w-full rounded-3xl bg-black"
                key={singleVideoUrl}
              >
                <source src={singleVideoUrl} type="video/mp4" />
              </video>

              <div className="mt-4 flex gap-3">
                <a
                  href={singleVideoUrl}
                  download
                  className="inline-flex items-center gap-2 rounded-full bg-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-sky-400"
                >
                  <FiDownload />
                  Download Clip
                </a>

                <button
                  type="button"
                  onClick={() => {
                    window.open(singleVideoUrl, "_blank");
                  }}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/80 px-5 py-3 text-sm font-semibold text-slate-100"
                >
                  <FiPlay />
                  Preview
                </button>
              </div>
            </div>
          ) : null}
        </>
      )}
    </motion.div>
  );
}
