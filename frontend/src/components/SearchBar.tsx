import { motion } from "framer-motion";
import { FiSearch } from "react-icons/fi";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  disabled?: boolean;
}

export default function SearchBar({ value, onChange, onSearch, disabled }: SearchBarProps) {
  return (
    <motion.div className="glass-card flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex-1">
        <label htmlFor="search-query" className="mb-2 block text-sm uppercase tracking-[0.25em] text-slate-400">
          Natural language search
        </label>
        <div className="relative">
          <FiSearch className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
          <input
            id="search-query"
            type="text"
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Ask something about the uploaded video"
            className="w-full rounded-3xl border border-white/10 bg-slate-950/90 py-4 pl-12 pr-4 text-slate-100 outline-none transition focus:border-sky-400/90 focus:ring-2 focus:ring-sky-400/20"
          />
        </div>
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={onSearch}
        className="mt-3 inline-flex items-center justify-center rounded-3xl bg-sky-500 px-6 py-4 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50 sm:mt-0"
      >
        Search
      </button>
    </motion.div>
  );
}
