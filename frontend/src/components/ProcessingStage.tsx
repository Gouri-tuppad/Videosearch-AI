interface ProcessingStageProps {
  step: string;
  status: string;
  message?: string | null;
}

export default function ProcessingStage({ step, status, message }: ProcessingStageProps) {
  const badge = status === "completed" ? "bg-emerald-500/15 text-emerald-300" : status === "running" ? "bg-sky-500/15 text-sky-200" : "bg-rose-500/15 text-rose-200";
  return (
    <div className="flex items-start gap-4 rounded-3xl border border-white/10 bg-surface px-4 py-4">
      <div className={`mt-1 h-3.5 w-3.5 rounded-full ${badge}`} />
      <div>
        <p className="text-sm font-semibold text-white">{step.replace(/_/g, " ")}</p>
        {message ? <p className="mt-1 text-sm text-slate-400">{message}</p> : null}
      </div>
    </div>
  );
}
