import http from 'http';
import fs from 'fs';
import path from 'path';

const PORT = 3000;

http.createServer((req, res) => {
    let filePath = '.' + req.url;
    if (filePath === './') filePath = './index.html';

    let extname = path.extname(filePath);
    let contentType = 'text/html';
    if (extname === '.js') contentType = 'text/javascript';
    if (extname === '.css') contentType = 'text/css';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            res.writeHead(500);
            res.end(`Error: ${error.code}`);
        } else {
            res.writeHead(200, {
                'Content-Type': contentType,
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp'
            });
            res.end(content, 'utf-8');
        }
    });
}).listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}/`);
});
