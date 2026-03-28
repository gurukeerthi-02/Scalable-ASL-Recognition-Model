const fs = require('fs');
const path = require('path');

const files = [
  'app/page.tsx',
  'app/join/page.tsx',
  'app/create/page.tsx',
  'app/call/[roomId]/page.tsx',
  'app/layout.tsx'
];

files.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');

  // Replace all ASL Meet text with Voiceout
  content = content.replace(/ASL Meet/g, 'Voiceout');

  // Update outer logo container in join and create to match homepage
  content = content.replace(
    /<div className="bg-palette-dark p-2 rounded-xl shadow-lg">\s*<Hand className="w-5 h-5 text-palette-medium" \/>\s*<\/div>/g,
    `<div className="bg-gradient-to-br from-palette-medium to-palette-light p-2 rounded-xl shadow-lg shadow-palette-medium/30">\n            <Hand className="w-5 h-5 text-palette-offwhite" />\n          </div>`
  );

  // Update text span in join and create to match homepage styling (font-extrabold tracking-wide)
  content = content.replace(
    /<span className="text-palette-dark">Voiceout<\/span>/g,
    `<span className="text-palette-dark font-extrabold tracking-wide">Voiceout</span>`
  );

  // Update outer logo container in call/[roomId] to match homepage (but keeping responsive sizing)
  content = content.replace(
    /<div className="p-1.5 bg-palette-dark rounded-lg md:rounded-xl flex-shrink-0">\s*<Hand className="w-4 h-4 md:w-5 md:h-5 text-palette-medium" \/>\s*<\/div>/g,
    `<div className="bg-gradient-to-br from-palette-medium to-palette-light p-1.5 rounded-lg md:rounded-xl flex-shrink-0 shadow-lg shadow-palette-medium/30">\n                <Hand className="w-4 h-4 md:w-5 md:h-5 text-palette-offwhite" />\n              </div>`
  );

  // Update text span in call/[roomId]
  content = content.replace(
    /<span className="hidden sm:block text-base md:text-lg font-bold text-palette-dark">Voiceout<\/span>/g,
    `<span className="hidden sm:block text-base md:text-lg font-extrabold tracking-wide text-palette-dark">Voiceout</span>`
  );

  fs.writeFileSync(file, content);
});

console.log('Logo and trademark updated');
