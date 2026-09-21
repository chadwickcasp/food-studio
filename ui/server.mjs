import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { foodStudioApi } from "./server/api.mjs";

const uiRoot = path.dirname(fileURLToPath(import.meta.url));
const distRoot = path.join(uiRoot, "dist");
const port = Number(process.env.PORT ?? 4173);

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = http.createServer((req, res) => {
  foodStudioApi(req, res, async () => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const requested = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    let target = path.resolve(distRoot, requested);
    if (!target.startsWith(`${distRoot}${path.sep}`) && target !== path.join(distRoot, "index.html")) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    try {
      const stat = await fs.stat(target);
      if (stat.isDirectory()) target = path.join(target, "index.html");
      const data = await fs.readFile(target);
      res.writeHead(200, { "content-type": mimeTypes[path.extname(target)] ?? "application/octet-stream" });
      res.end(data);
    } catch {
      const data = await fs.readFile(path.join(distRoot, "index.html"));
      res.writeHead(200, { "content-type": mimeTypes[".html"] });
      res.end(data);
    }
  });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Food Studio Review running at http://127.0.0.1:${port}`);
});
