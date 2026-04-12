import logoImg from "../../imports/ZotAssistantLogo.png";

interface EmptyStateProps {
  onSuggestionClick: (suggestion: string) => void;
}

const suggestions = [
  "What are the prereqs for COMPSCI 161?",
  "What are the CS major requirements?",
  "What is the academic integrity policy?",
];

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6">
      <div className="mb-6">
        <img
          src={logoImg}
          alt="ZotAssistant Logo"
          className="h-20 w-auto object-contain"
        />
      </div>

      <h1 className="text-[#1A1A1A] font-bold text-2xl mb-3 text-center">
        Ask me anything about UCI courses & policies
      </h1>

      <p className="text-[#6B7280] text-center mb-8 max-w-md">
        Get help with prerequisites, academic policies, major requirements, and
        more.
      </p>

      <div className="flex flex-wrap gap-2 md:gap-3 justify-center max-w-2xl">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion)}
            className="px-3 md:px-4 py-2 md:py-2.5 border-2 border-[#0064A4] text-[#0064A4] rounded-full hover:bg-[#0064A4] hover:text-white transition-colors duration-200 text-xs md:text-sm font-medium"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
