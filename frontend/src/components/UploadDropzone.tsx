import { motion } from "framer-motion";
import type { DragEvent } from "react";
import { FiUploadCloud } from "react-icons/fi";

interface UploadDropzoneProps {
  isDragging: boolean;
  selectedFileName?: string | null;
  onSelectFile: (file: File) => void;
  onDragStateChange: (dragging: boolean) => void;
}

export default function UploadDropzone({ isDragging, selectedFileName, onSelectFile, onDragStateChange }: UploadDropzoneProps) {
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onDragStateChange(false);

    const file = event.dataTransfer.files?.[0];
    if (file) {
      onSelectFile(file);
    }
  };

  return (
    <motion.label
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      htmlFor="video-upload"
      onDragEnter={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onDragStateChange(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onDragStateChange(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onDragStateChange(false);
      }}
      onDrop={handleDrop}
      className={`glass-card flex min-h-[220px] flex-col items-center justify-center gap-4 border-dashed border-white/15 px-6 py-8 text-center transition-all duration-300 ${
        isDragging ? "border-sky-400/70 bg-slate-900/70" : "bg-surface"
      }`}
    >
      <FiUploadCloud className="h-12 w-12 text-sky-300" />
      <div>
        <p className="text-lg font-semibold text-slate-100">Drag & drop a video file</p>
        <p className="mt-1 text-sm text-slate-400">or click to choose a file for AI analysis.</p>
      </div>
      <p className="max-w-xs text-sm text-slate-400">
        {selectedFileName ? `Selected file: ${selectedFileName}` : "No file selected yet."}
      </p>
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onSelectFile(file);
          }
          event.target.value = "";
        }}
      />
    </motion.label>
  );
}
