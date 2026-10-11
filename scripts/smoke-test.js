#!/usr/bin/env node
// End-to-end smoke test. Run before any deploy.
//
// What it does:
//   1. Hits /api/health — confirms the server is up and (optionally) LM Studio
//      is reachable.
//   2. POSTs a URL to /api/process — confirms the submit pipeline accepts it.
//   3. Polls /api/status/:jobId until done or timeout.
//   4. Fetches the reader.html — confirms output renders with critical markers
//      (phrase spans, flashcard save button).
//   5. Exits 0 on success, 1 on any failure.
//
// Usage:
//   node scripts/smoke-test.js                    # uses defaults below
//   node scripts/smoke-test.js --url https://...  # custom test URL
//   SMOKE_URL=https://... npm run test:smoke
//   SMOKE_SERVER=http://prod.example.com node scripts/smoke-test.js
//
// Notes:
//   - Requires the server to be running with TURNSTILE_SECRET_KEY UNSET
//     (dev/test mode — Turnstile is bypassed). Production smoke tests should
//     either run from a permitted IP or pre-fetch a real Turnstile token.
//   - The default test URL should be one you have a recent cache hit for so
//     the test runs in seconds, not minutes. Override per environment.

const SERVER = process.env.SMOKE_SERVER || argFor('--server') || 'http://localhost:3000';
const TEST_URL = process.env.SMOKE_URL || argFor('--url') || ''; // no default — user must set
const TIMEOUT_MS = parseInt(process.env.SMOKE_TIMEOUT_MS || '120000', 10);

function argFor(flag) {
  const i = process.argv.indexOf(flag);
  return i >= 0 ? process.argv[i + 1] : null;
}

let exitCode = 0;
function pass(label) { console.log(`  PASS: ${label}`); }
function fail(label, detail) { console.log(`  FAIL: ${label}${detail ? ' — ' + detail : ''}`); exitCode = 1; }

async function main() {
  if (!TEST_URL) {
    console.error('SMOKE_URL or --url is required (a chapter URL the server can process).');
    console.error('Tip: pick a URL you have a recent cache hit for so this test runs in seconds.');
    process.exit(2);
  }

  console.log(`Smoke test → server=${SERVER}, url=${TEST_URL}`);

  // Step 1: health
  console.log('\n[1/4] Health check');
  try {
    const r = await fetch(`${SERVER}/api/health`);
    if (!r.ok) { fail('GET /api/health', `HTTP ${r.status}`); return finish(); }
    const body = await r.json();
    if (body.ok !== true) { fail('health body.ok', JSON.stringify(body)); return finish(); }
    pass(`server up (uptime=${body.uptimeSeconds}s, version=${body.version})`);
    if (body.lmStudioReachable === false) {
      console.log('  WARN: LM Studio not reachable — uncached submissions will fail');
    } else {
      pass('LM Studio reachable');
    }
  } catch (e) {
    fail('GET /api/health', e.message); return finish();
  }

  // Step 2: submit
  console.log('\n[2/4] Submit chapter');
  let jobId;
  try {
    const r = await fetch(`${SERVER}/api/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: TEST_URL, level: 'hsk3' }),
    });
    const body = await r.json();
    if (!r.ok) { fail('POST /api/process', `HTTP ${r.status} ${JSON.stringify(body)}`); return finish(); }
    if (!body.jobs || !body.jobs[0] || !body.jobs[0].jobId) {
      fail('POST /api/process body shape', JSON.stringify(body)); return finish();
    }
    jobId = body.jobs[0].jobId;
    pass(`accepted, jobId=${jobId} fromCache=${!!body.jobs[0].fromCache}`);
  } catch (e) {
    fail('POST /api/process', e.message); return finish();
  }

  // Step 3: poll
  console.log('\n[3/4] Poll status');
  const start = Date.now();
  let resultUrl = null;
  while (Date.now() - start < TIMEOUT_MS) {
    await new Promise(r => setTimeout(r, 2000));
    try {
      const r = await fetch(`${SERVER}/api/status/${jobId}`);
      if (!r.ok) { fail(`GET /api/status/${jobId}`, `HTTP ${r.status}`); return finish(); }
      const body = await r.json();
      if (body.status === 'complete') {
        resultUrl = body.resultUrl;
        pass(`completed in ${Math.round((Date.now() - start) / 1000)}s`);
        break;
      }
      if (body.status === 'error') {
        fail('job ended in error', `${body.errorCode || ''} ${body.error || ''}`); return finish();
      }
      // still pending/downloading/processing — keep polling
      process.stdout.write('.');
    } catch (e) {
      fail('poll', e.message); return finish();
    }
  }
  if (!resultUrl) {
    fail('poll', `timed out after ${TIMEOUT_MS / 1000}s`); return finish();
  }

  // Step 4: fetch reader.html and assert markers
  console.log('\n[4/4] Verify reader.html');
  try {
    const r = await fetch(`${SERVER}${resultUrl}`);
    if (!r.ok) { fail(`GET ${resultUrl}`, `HTTP ${r.status}`); return finish(); }
    const html = await r.text();
    if (!html.includes('data-pinyin=')) fail('phrase spans (data-pinyin)');
    else pass('phrase spans present');
    if (!html.includes('def-save')) fail('flashcard save button (def-save)');
    else pass('flashcard save button present');
    if (!html.includes('switchLevel')) fail('HSK level toggle (switchLevel)');
    else pass('HSK level toggle present');
  } catch (e) {
    fail('fetch reader.html', e.message); return finish();
  }

  finish();
}

function finish() {
  console.log(`\n${exitCode === 0 ? 'OK — smoke test passed' : 'FAIL — smoke test failed'}`);
  process.exit(exitCode);
}

main().catch(e => { console.error(e); process.exit(1); });
