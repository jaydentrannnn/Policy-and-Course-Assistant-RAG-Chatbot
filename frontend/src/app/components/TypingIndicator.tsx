import { motion } from "motion/react";

export function TypingIndicator() {
  return (
    <div className="flex justify-start mb-6">
      <div className="bg-white border-l-[3px] border-[#FFD200] rounded-[18px_18px_18px_4px] px-4 py-3 shadow-sm">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 bg-[#6B7280] rounded-full"
              animate={{
                y: [0, -6, 0],
                opacity: [0.4, 1, 0.4],
              }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                delay: i * 0.15,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
