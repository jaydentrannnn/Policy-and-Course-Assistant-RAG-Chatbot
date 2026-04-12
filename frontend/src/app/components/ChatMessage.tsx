import { motion } from "motion/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Known suffix replacements for degree designations in URL slugs
const _SUFFIX_MAP: Record<string, string> = {
  Bs: "B.S.",
  Ba: "B.A.",
  Ms: "M.S.",
  Phd: "Ph.D.",
  Minor: "Minor",
};

// Generic path segments that carry no useful page identity on their own —
// fall back to the segment before them for a more meaningful label.
const _GENERIC_SEGMENTS = new Set([
  "allcourses",
  "index",
  "home",
  "undergraduate",
  "graduate",
  "requirements",
  "overview",
]);

function _titleCase(slug: string): string {
  return slug
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function _cleanSegment(raw: string): string {
  // Title-case first, then apply suffix map for degree designations
  const titled = _titleCase(raw);
  return titled
    .split(" ")
    .map((word) => _SUFFIX_MAP[word] ?? word)
    .join(" ");
}

function labelFromUrl(href: string): string {
  try {
    const url = new URL(href);
    const { hostname, pathname } = url;

    // Determine domain prefix
    let prefix = "UCI";
    if (hostname === "catalogue.uci.edu") prefix = "Catalogue";
    else if (hostname.includes("reg.uci.edu")) prefix = "Registrar";
    else if (hostname.includes("conduct.uci.edu")) prefix = "Student Conduct";

    // Pick a meaningful path segment — skip trailing generic segments
    const segments = pathname.split("/").filter(Boolean);
    let label = "";
    for (let i = segments.length - 1; i >= 0; i--) {
      if (!_GENERIC_SEGMENTS.has(segments[i].toLowerCase())) {
        label = _cleanSegment(segments[i]);
        break;
      }
    }

    return label ? `${prefix} · ${label}` : prefix;
  } catch {
    return "Source";
  }
}

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-6`}
    >
      <div
        className={`min-w-0 max-w-[90%] ${isUser ? "md:max-w-[70%]" : "md:max-w-[85%]"}`}
      >
        <div
          className={`${
            isUser
              ? "bg-[#0064A4] text-white rounded-[18px_18px_4px_18px]"
              : "bg-white border-l-[3px] border-[#FFD200] rounded-[18px_18px_18px_4px]"
          } px-4 py-3 shadow-sm overflow-hidden`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{content}</p>
          ) : (
            <div className="prose prose-sm max-w-none prose-a:text-[#0064A4] prose-a:underline prose-a:break-all prose-p:text-[#1A1A1A] prose-headings:text-[#1A1A1A] prose-li:text-[#1A1A1A] prose-strong:text-[#1A1A1A]">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a({ href, children }) {
                    // When the link text is the bare URL (autolinked by remark-gfm),
                    // replace it with a human-readable label derived from the URL.
                    const isBareUrl = href && String(children) === href;
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[#0064A4] underline font-medium"
                      >
                        {isBareUrl ? labelFromUrl(href) : children}
                      </a>
                    );
                  },
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        <div
          className={`text-xs text-[#6B7280] mt-1 ${
            isUser ? "text-right" : "text-left"
          }`}
        >
          {timestamp}
        </div>
      </div>
    </motion.div>
  );
}
