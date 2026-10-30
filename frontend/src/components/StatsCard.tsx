interface StatsCardProps {
  label: string;
  value: string;
  accent?: string;
}

export default function StatsCard({ label, value, accent = "bg-sky-500/15 text-sky-200" }: StatsCardProps) {
  return (
    <div className="glass-card rounded-3xl p-5">
      <p className="text-sm uppercase tracking-[0.24em] text-slate-400">{label}</p>
      <p className={`mt-4 text-3xl font-semibold ${accent}`}>{value}</p>
    </div>
  );
}
