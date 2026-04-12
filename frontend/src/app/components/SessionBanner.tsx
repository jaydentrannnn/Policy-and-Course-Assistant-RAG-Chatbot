import { X } from "lucide-react";
import { motion } from "motion/react";

interface SessionBannerProps {
  onDismiss: () => void;
}

export function SessionBanner({ onDismiss }: SessionBannerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-[#FFF8E1] border-l-[3px] border-[#FFD200] px-4 py-3 flex items-center justify-between"
    >
      <p className="text-xs md:text-sm text-[#1A1A1A] pr-2">
        This is a new session. Your conversation will not be saved after you
        close or refresh this page.
      </p>
      <button
        onClick={onDismiss}
        className="ml-4 text-[#6B7280] hover:text-[#1A1A1A] transition-colors focus:outline-none focus:ring-2 focus:ring-[#FFD200] rounded"
        aria-label="Dismiss banner"
      >
        <X size={18} />
      </button>
    </motion.div>
  );
}
