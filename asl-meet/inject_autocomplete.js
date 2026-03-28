const fs = require('fs');

let aslTsx = fs.readFileSync('components/video/ASLIndicator.tsx', 'utf8');

// Add imports
if (!aslTsx.includes("import { COMMON_WORDS }")) {
  aslTsx = aslTsx.replace("import { Card } from '@/components/ui/card';", "import { useMemo } from 'react';\nimport { Card } from '@/components/ui/card';\nimport { COMMON_WORDS } from '@/lib/dictionary';");
}

// Add onSelectSuggestion to interface
if (!aslTsx.includes('onSelectSuggestion?:')) {
  aslTsx = aslTsx.replace(
    'onBackspace?: () => void;',
    'onBackspace?: () => void;\n  onSelectSuggestion?: (word: string) => void;'
  );
}

// Add onSelectSuggestion to destructured props
if (!aslTsx.includes('onSelectSuggestion,')) {
  aslTsx = aslTsx.replace(
    'onBackspace,',
    'onBackspace,\n  onSelectSuggestion,'
  );
}

// Add the suggestions memo hook right inside the component start
const componentStart = 'export function ASLIndicator({';
const returnStart = '  return (';
const hookInjection = `\n  const suggestions = useMemo(() => {
    const words = sentenceBuffer.split(' ');
    const lastWord = words[words.length - 1].toUpperCase();
    if (!lastWord || lastWord.length < 2) return [];
    
    const matches = COMMON_WORDS.filter(w => 
      w.toUpperCase().startsWith(lastWord) && w.toUpperCase() !== lastWord
    );
    return matches.slice(0, 3).map(w => w.toUpperCase());
  }, [sentenceBuffer]);

`;

if (!aslTsx.includes('const suggestions = useMemo')) {
  aslTsx = aslTsx.replace(returnStart, hookInjection + returnStart);
}

// Now replace Current Sentence header to include suggestions flexbox
const headerRegex = /<h4 className="text-\[10px\] font-bold uppercase tracking-wider text-palette-dark\/60 ml-1">Current Sentence<\/h4>/;
const newHeader = `<div className="flex items-center justify-between ml-1">
                <h4 className="text-[10px] font-bold uppercase tracking-wider text-palette-dark/60">Current Sentence</h4>
                <div className="flex gap-1.5 h-4">
                  {suggestions.map((sug) => (
                    <button
                      key={sug}
                      onClick={() => onSelectSuggestion?.(sug)}
                      className="px-2 py-0.5 bg-palette-light/30 hover:bg-palette-medium hover:text-white text-palette-dark text-[9px] font-bold rounded shadow-sm border border-palette-medium/30 transition-all active:scale-95 flex items-center"
                    >
                      {sug}
                    </button>
                  ))}
                </div>
              </div>`;

if (aslTsx.match(headerRegex)) {
  aslTsx = aslTsx.replace(headerRegex, newHeader);
}

fs.writeFileSync('components/video/ASLIndicator.tsx', aslTsx);
console.log('ASLIndicator updated with autocomplete logic!');

// Patching page.tsx
let pageTsx = fs.readFileSync('app/call/[roomId]/page.tsx', 'utf8');

// 1. Add handleSelectSuggestion hook
const handleSelectSuggestionHook = `
  const handleSelectSuggestion = useCallback((word: string) => {
    setSentenceBuffer(prev => {
      const words = prev.split(' ');
      words.pop(); // remove partial word
      words.push(word);
      return words.join(' ') + ' '; // add completed word with a trailing space!
    });
  }, []);
`;

const handleBackspaceRegex = /const handleBackspace = useCallback[\s\S]*?\}, \[\]\);/;
const matchHook = pageTsx.match(handleBackspaceRegex);

if (matchHook && !pageTsx.includes('handleSelectSuggestion')) {
  pageTsx = pageTsx.replace(handleBackspaceRegex, matchHook[0] + handleSelectSuggestionHook);
}

// 2. Pass handleSelectSuggestion to ASLIndicator usages
if (!pageTsx.includes('onSelectSuggestion=')) {
  pageTsx = pageTsx.replace(/onBackspace=\{handleBackspace\}/g, 'onBackspace={handleBackspace}\n                    onSelectSuggestion={handleSelectSuggestion}');
}

fs.writeFileSync('app/call/[roomId]/page.tsx', pageTsx);
console.log('page.tsx handler successfully injected!');
