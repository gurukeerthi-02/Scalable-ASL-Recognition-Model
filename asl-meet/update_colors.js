const fs = require('fs');

const files = [
  'app/join/page.tsx',
  'app/create/page.tsx',
  'app/call/[roomId]/page.tsx',
  'components/video/VideoTile.tsx',
  'components/video/CallControls.tsx',
  'components/video/ASLPanel.tsx',
  'components/video/ASLIndicator.tsx',
];

files.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');
  
  // Background Gradients
  content = content.replace(/bg-gradient-to-br from-yellow-400 via-yellow-300 to-yellow-400/g, 'bg-palette-offwhite');
  content = content.replace(/from-yellow-400 via-yellow-300 to-yellow-400/g, 'bg-palette-offwhite');
  content = content.replace(/bg-gradient-to-br from-yellow-400 to-yellow-500/g, 'bg-gradient-to-br from-palette-medium to-palette-light');
  
  // Yellow classes
  content = content.replace(/yellow-400\/10/g, 'palette-light/30');
  content = content.replace(/yellow-400\/20/g, 'palette-light/40');
  content = content.replace(/yellow-400\/30/g, 'palette-light/50');
  content = content.replace(/yellow-400\/50/g, 'palette-light/60');
  
  // Standard replacements
  content = content.replace(/yellow-400/g, 'palette-medium');
  content = content.replace(/yellow-300/g, 'palette-light');
  content = content.replace(/yellow-500/g, 'palette-medium');
  
  // Replace references with "black" to our palette dark color to maintain theme consistency
  content = content.replace(/bg-black\/5/g, 'bg-palette-dark/5');
  content = content.replace(/bg-black\/10/g, 'bg-palette-dark/10');
  content = content.replace(/bg-black\/90/g, 'bg-palette-dark/90');
  content = content.replace(/bg-black\/70/g, 'bg-palette-dark/70');
  
  content = content.replace(/text-black\/80/g, 'text-palette-dark/80');
  content = content.replace(/text-black\/70/g, 'text-palette-dark/70');
  content = content.replace(/text-black\/60/g, 'text-palette-dark/60');
  content = content.replace(/text-black\/50/g, 'text-palette-dark/50');
  content = content.replace(/text-black\/40/g, 'text-palette-dark/40');
  
  content = content.replace(/shadow-black\/20/g, 'shadow-palette-dark/20');
  content = content.replace(/border-black\/30/g, 'border-palette-dark/30');
  content = content.replace(/border-black\/10/g, 'border-palette-dark/10');

  // Fix button styles that were `bg-black text-yellow-400` which maps to `bg-black text-palette-medium` now
  // We want `bg-palette-dark text-palette-offwhite`
  content = content.replace(/bg-black(.*?)text-palette-medium/g, 'bg-palette-dark$1text-palette-offwhite');
  content = content.replace(/text-palette-medium(.*?)bg-black/g, 'text-palette-offwhite$1bg-palette-dark');
  content = content.replace(/text-palette-medium(.*?)bg-palette-dark/g, 'text-palette-offwhite$1bg-palette-dark');
  content = content.replace(/text-palette-medium(.*?)bg-black\/90/g, 'text-palette-offwhite$1bg-palette-dark');
  
  // For file VideoTile specifically, we might want to keep the inner tile black.
  if (file === 'components/video/VideoTile.tsx') {
    content = content.replace(/bg-black(?![\w\-\/])/g, 'bg-slate-900'); // Video background looks better slate-900 than pure palette-dark.
    content = content.replace(/text-black(?![\w\-\/])/g, 'text-palette-dark');
  } else if (file === 'components/video/CallControls.tsx') {
     // Tooltips and buttons
     content = content.replace(/bg-black/g, 'bg-palette-dark');
     content = content.replace(/text-black/g, 'text-palette-offwhite'); // for contrast with dark
     content = content.replace(/border-black/g, 'border-palette-dark');
  } else {
    // General replacements ensuring no trailing extensions
    content = content.replace(/bg-black(?![\w\-\/])/g, 'bg-palette-dark');
    content = content.replace(/border-black(?![\w\-\/])/g, 'border-palette-dark');
    // text-black can be text-palette-dark
    content = content.replace(/text-black(?![\w\-\/])/g, 'text-palette-dark');
  }

  // One final sweep over palette-medium text in palette-dark areas matching CallControls logic
  content = content.replace(/bg-palette-dark(.*?)text-palette-medium/g, 'bg-palette-dark$1text-palette-offwhite');

  fs.writeFileSync(file, content);
});
console.log('Update complete');
