const fs = require('fs');

let pageTsx = fs.readFileSync('app/call/[roomId]/page.tsx', 'utf8');

// Find handleAddSpace block
const handleSpaceRegex = /const handleAddSpace = useCallback\([^)]+\) => \{[\s\S]*?\}, \[\]\);/;
const matchHook = pageTsx.match(handleSpaceRegex);

if (matchHook && !pageTsx.includes('const handleBackspace')) {
  const backspaceHook = `\n\n  const handleBackspace = useCallback(() => {\n    setSentenceBuffer(prev => prev.slice(0, -1));\n  }, []);`;
  pageTsx = pageTsx.replace(handleSpaceRegex, matchHook[0] + backspaceHook);
  fs.writeFileSync('app/call/[roomId]/page.tsx', pageTsx);
  console.log('Fixed handleBackspace hook');
} else {
  console.log('Already fixed or not found');
}
