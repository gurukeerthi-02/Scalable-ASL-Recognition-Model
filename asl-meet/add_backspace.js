const fs = require('fs');

// Patching ASLIndicator.tsx
let aslIndicator = fs.readFileSync('components/video/ASLIndicator.tsx', 'utf8');

// 1. Add Delete icon
aslIndicator = aslIndicator.replace(/Trash2, Space, Zap/, 'Trash2, Space, Delete, Zap');

// 2. Add onBackspace to interface
aslIndicator = aslIndicator.replace(
  /onAddSpace\?: \(\) => void;/g, 
  'onAddSpace?: () => void;\n  onBackspace?: () => void;'
);

// 3. Add onBackspace to function props
aslIndicator = aslIndicator.replace(
  /onAddSpace,\s+receivedMessages/g,
  'onAddSpace,\n  onBackspace,\n  receivedMessages'
);

// 4. Add the button itself
const addSpaceBtnRegex = /<Button\s+size="icon"\s+variant="secondary"\s+onClick=\{onAddSpace\}[\s\S]*?<Space className="w-4 h-4" \/>\s*<\/Button>/;
const matchBtn = aslIndicator.match(addSpaceBtnRegex);
if(matchBtn) {
  const backspaceBtn = `<Button
                    size="icon"
                    variant="secondary"
                    onClick={onBackspace}
                    disabled={!sentenceBuffer.length}
                    className="h-9 w-9 bg-white text-palette-dark border border-gray-200 hover:bg-orange-500 hover:border-orange-500 hover:text-white rounded-xl shadow-sm transition-all disabled:opacity-30"
                    title="Backspace"
                  >
                    <Delete className="w-4 h-4" />
                  </Button>\n                  ` + matchBtn[0];
  aslIndicator = aslIndicator.replace(addSpaceBtnRegex, backspaceBtn);
}
fs.writeFileSync('components/video/ASLIndicator.tsx', aslIndicator);

// Patching page.tsx
let pageTsx = fs.readFileSync('app/call/[roomId]/page.tsx', 'utf8');

// 1. Add handleBackspace hook
const handleAddSpaceHook = `  const handleAddSpace = useCallback(() => {
    setSentenceBuffer(prev => prev + ' ');
    // Remove lastRecognizedLetterRef clearing to prevent re-adding current sign
  }, []);`;
const handleBackspaceHook = `${handleAddSpaceHook}\n\n  const handleBackspace = useCallback(() => {\n    setSentenceBuffer(prev => prev.slice(0, -1));\n  }, []);`;
pageTsx = pageTsx.replace(handleAddSpaceHook, handleBackspaceHook);

// 2. Pass handleBackspace to ASLIndicator (which happens twice usually: inline mode and panel mode)
pageTsx = pageTsx.replace(/onAddSpace=\{handleAddSpace\}/g, 'onAddSpace={handleAddSpace}\n                    onBackspace={handleBackspace}');

fs.writeFileSync('app/call/[roomId]/page.tsx', pageTsx);
console.log('Backspace successfully added!');
