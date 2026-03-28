const https = require('https');
const fs = require('fs');
const path = require('path');

const url = 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt';
const outputPath = path.join(__dirname, 'lib', 'dictionary.ts');

https.get(url, (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    // Only take top 3000 to keep bundle small but useful
    const words = data.split('\n').filter(w => w.length > 2).slice(0, 3000);
    const tsCode = `export const COMMON_WORDS: string[] = ${JSON.stringify(words)};\n`;
    fs.writeFileSync(outputPath, tsCode);
    console.log('Dictionary generated successfully at lib/dictionary.ts');
  });
}).on('error', (err) => {
  console.error('Error fetching dictionary:', err);
});
