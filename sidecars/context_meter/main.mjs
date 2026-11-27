/**
 * Context Meter Sidecar Backend.
 * Serves the real-time context usage dashboard and Context Guard linkage APIs.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Response, SidecarApp } from 'sidecar_sdk';
import { buildContextSnapshot, findAppDataDirs, getGuardDataDir, ID_RE } from './meter.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC_ASSETS = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/index.html': ['index.html', 'text/html; charset=utf-8'],
  '/styles.css': ['styles.css', 'text/css; charset=utf-8'],
  '/app.js': ['app.js', 'application/javascript; charset=utf-8'],
};

const app = new SidecarApp();
const transcriptCache = new Map();

for (const [route, [filename, contentType]] of Object.entries(STATIC_ASSETS)) {
  const filePath = join(__dirname, filename);
  app.page(route, () => new Response(readFileSync(filePath, 'utf-8'), { contentType }));
}

app.api('/api/context', (data) => {
  const convId = String(data?.conversationId || app.config.conversationId || '').trim();
  if (!convId || !ID_RE.test(convId)) {
    return { conversationId: null, found: false, reason: 'no_conversation' };
  }
  const snap = buildContextSnapshot({
    convId,
    appDataDirs: findAppDataDirs(),
    guardDataDir: getGuardDataDir(),
    cache: transcriptCache,
  });
  if (snap.found && data?.etag && data.etag === snap.etag) {
    return { conversationId: convId, found: true, unchanged: true, etag: snap.etag };
  }
  return snap;
}, 'GET');

app.run().then(() => {
  console.log(`[context_meter] ready; appDataDirs=${findAppDataDirs().join(', ') || '(none)'}`);
});
