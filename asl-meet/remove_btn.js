const fs = require('fs');
let content = fs.readFileSync('app/call/[roomId]/page.tsx', 'utf8');

const targetStr = `            <Button
              onClick={handleLeave}
              className="rounded-full px-4 md:px-6 h-9 md:h-10 font-bold bg-red-600 hover:bg-red-700 text-white shadow-lg"
            >
              <span className="hidden sm:inline">Leave</span>
              <span className="sm:hidden">End</span>
            </Button>`;

// Replace with regex ignoring exact whitespace
const strRegex = /<Button\s+onClick=\{handleLeave\}\s+className="rounded-full[^>]+>\s*<span[^>]+>Leave<\/span>\s*<span[^>]+>End<\/span>\s*<\/Button>/;

content = content.replace(strRegex, '');

fs.writeFileSync('app/call/[roomId]/page.tsx', content);
console.log('Button removed');
