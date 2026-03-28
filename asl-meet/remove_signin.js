const fs = require('fs');
let content = fs.readFileSync('app/page.tsx', 'utf8');

const strRegex = /<Button\s+size="sm"[\s\S]*?onClick=\{\(\) => router\.push\('\/join'\)\}[\s\S]*?className="border-2[^>]+>\s*Sign In\s*<\/Button>/;

content = content.replace(strRegex, '');

fs.writeFileSync('app/page.tsx', content);
console.log('SignIn Button removed');
