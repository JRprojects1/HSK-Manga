#!/usr/bin/env node

import * as dotenv from 'dotenv';
import * as path from 'path';
import * as crypto from 'crypto';

// Load .env file from project root
dotenv.config({ path: path.join(__dirname, '..', '.env') });

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import compression from 'compression';
import cookieParser from 'cookie-parser';
import apiRoutes from './routes/api';
import authRoutes from './routes/auth';
import { rehydrateCompletedJobs } from './services/jobService';
import { initLogService } from './services/logService';
import { writeWeeklySnapshot } from './services/metricsSnapshotService';

const app = express();
const PORT = process.env.PORT || 3000;

// Trust the first reverse proxy (Cloudflare Tunnel). Without this, req.ip is
// the local Node socket and per-IP rate limiting wouldn't actually distinguish
// users. We additionally check `cf-connecting-ip` in the API layer for safety.
app.set('trust proxy', 1);

// Compression first, before any body parsing.
app.use(compression({
  filter: (req, res) => {
    const type = res.getHeader('Content-Type') as string || '';
    if (type.startsWith('image/')) return false;
    return compression.filter(req, res);
  }
}));
app.use(cors());

app.use(express.json());
app.use(cookieParser());

// Set an anonymous browser_id cookie on first visit. Used by
// submissionHistoryService for per-browser analytics. One year expiry,
// HttpOnly so JS can't read it.
app.use((req: Request, res: Response, next: NextFunction) => {
  if (!req.cookies?.browser_id) {
    const id = crypto.randomUUID();
    res.cookie('browser_id', id, {
      httpOnly: true,
      sameSite: 'lax',
      secure: true,
      maxAge: 365 * 24 * 60 * 60 * 1000,
    });
    // Make the new id available to downstream handlers within the SAME request
    // (res.cookie only sends to the browser; req.cookies wouldn't see it
    // otherwise until the next request).
    req.cookies = { ...req.cookies, browser_id: id };
  }
  next();
});

// Serve frontend static files
app.use(express.static(path.join(__dirname, '..', 'public')));

// Pretty URL for the admin dashboard (/admin → public/admin.html). The page is
// just a shell; its data is gated by ADMIN_TOKEN on /api/admin/stats.
app.get('/admin', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'admin.html'));
});

// Pretty URL for the feedback page (/feedback → public/feedback.html).
app.get('/feedback', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'feedback.html'));
});

// Pretty URL for the about page (/about → public/about.html).
app.get('/about', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'about.html'));
});

// Pretty URL for the chapter-processing form (/start → public/start.html).
// `/` is the marketing landing page (public/index.html, served by express.static).
app.get('/start', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'start.html'));
});

// Serve processed manga results
app.use('/outputs', express.static(path.join(__dirname, '..', 'outputs')));

// Shared word-audio library: citation clips are chapter-independent, so one
// store serves every chapter — and the CDN caches each clip once, site-wide.
// Seeded by scripts/seed-word-library.js, grown lazily by chapter builds.
app.use('/word-audio', express.static(
  process.env.WORD_LIBRARY_DIR || path.join(__dirname, '..', 'data', 'word-library'),
  { maxAge: '30d', immutable: true },  // content-addressed file names
));

// API routes. Auth routes mounted BEFORE apiRoutes so /api/auth/* and /api/me
// are handled by the auth router, not the general api router.
app.use('/api', authRoutes);
app.use('/api', apiRoutes);

// Phase 1.6: ensure logs/ exists, prune stale per-job logs
initLogService();

// Rehydrate completed jobs from disk on startup
rehydrateCompletedJobs();

// Auto weekly-metrics snapshot. Runs once now and every 24h thereafter, so the
// current ISO week's row in data/metrics-weekly.json stays fresh with no manual
// `npm run snapshot`. Idempotent per week; the committed evidence copy is pulled
// to Computer 1 via `npm run pull-metrics`. unref() so the timer never keeps the
// process alive on its own.
writeWeeklySnapshot();
const SNAPSHOT_INTERVAL_MS = 24 * 60 * 60 * 1000;
setInterval(() => writeWeeklySnapshot(), SNAPSHOT_INTERVAL_MS).unref();

// Start server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Press Ctrl+C to stop`);
});
