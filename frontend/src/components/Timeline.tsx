interface TimelineEvent {
  label: string;
  progress: number;
}

interface TimelineProps {
  events: TimelineEvent[];
}

export default function Timeline({ events }: TimelineProps) {
  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Interactive timeline</p>
          <h3 className="mt-2 text-xl font-semibold text-white">Search and processing journey</h3>
        </div>
      </div>
      <div className="mt-6 space-y-4">
        {events.map((event) => (
          <div key={event.label} className="space-y-2">
            <div className="flex items-center justify-between text-sm text-slate-400">
              <span>{event.label}</span>
              <span>{event.progress}%</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-white/5">
              <div className="h-full rounded-full bg-sky-500 transition-all" style={{ width: `${event.progress}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
