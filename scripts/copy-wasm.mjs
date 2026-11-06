// Copies the MediaPipe runtime into public/ so the app serves it itself
// (works offline and doesn't depend on a CDN).
import { cpSync, mkdirSync } from 'node:fs';
const src = 'node_modules/@mediapipe/tasks-vision/wasm';
const dst = 'public/mediapipe/wasm';
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log(`copied ${src} -> ${dst}`);
