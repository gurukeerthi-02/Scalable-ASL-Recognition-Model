const fs = require('fs');

let content = fs.readFileSync('app/call/[roomId]/page.tsx', 'utf8');

const regex = /const convertToSpeechText = useCallback\(\(text: string\) => \{[\s\S]*?return text;\n  \}, \[\]\);/;

const replacement = `const convertToSpeechText = useCallback((text: string) => {
    // TTS engines read ALL-CAPS strings by spelling out each letter as if it were an acronym.
    // Converting the entire input text to lowercase ensures the TTS speaks the words fluently 
    // as a normal sentence, negating the need for custom word mapping tables.
    return text.toLowerCase();
  }, []);`;

content = content.replace(regex, replacement);

fs.writeFileSync('app/call/[roomId]/page.tsx', content);
console.log('Fixed TTS logic mapping');
