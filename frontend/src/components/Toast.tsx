import { motion, AnimatePresence } from "framer-motion";

export type ToastKind = "success" | "error" | "info";

type ToastProps = {
  message: string;
  kind: ToastKind;
  visible: boolean;
};

const kindStyles: Record<ToastKind, string> = {
  success: "bg-emerald-500/95 text-slate-950",
  error: "bg-rose-500/95 text-white",
  info: "bg-sky-500/95 text-slate-950",
};

export default function Toast({ message, kind, visible }: ToastProps) {
  return (
    <AnimatePresence>
      {visible ? (
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 24 }}
          transition={{ duration: 0.25 }}
          role="status"
          aria-live="polite"
          className={`pointer-events-none fixed right-4 top-4 z-50 rounded-3xl px-5 py-3 text-sm font-semibold shadow-glass ${kindStyles[kind]}`}
        >
          {message}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
