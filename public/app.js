// ============================================================
// HSK Manga — staged-flow frontend (Start → Run, + Flashcards)
// ============================================================

// ---- Error code → user-facing message map ----
const ERRORS = {
  BAD_URL: () => ({ msg: "That doesn't look like a chapter URL. Paste the link from your browser's address bar (starts with http:// or https://).", target: 'url' }),
  NO_IMAGES_FOUND: () => ({ msg: "We loaded the page but couldn't find any chapter images. Make sure you pasted a single chapter URL, not the series homepage.", target: 'job' }),
  NO_READER: () => ({ msg: "Sorry, we couldn't get any readable images from this URL. Try a chapter from a different site.", target: 'job' }),
  TOO_MANY_PAGES: (m) => ({ msg: `This URL has ${m.pageCount || '?'} images — over the ${m.maxPages || '?'}-page limit. Submit a single chapter rather than an archive page.`, target: 'job' }),
  SITE_BLOCKED: () => ({ msg: "This site blocks automated downloads. Try a different reading source.", target: 'job' }),
  SITE_NOT_SUPPORTED: () => ({ msg: "This URL isn't supported. Try a chapter URL from a different webtoon or manga site.", target: 'url' }),
  QUOTA_EXCEEDED: (m) => ({ msg: `Daily limit reached (${m.limit?.max || ''} chapters per day). Try again in ${formatRetryAfter(m.retryAfterSeconds)}.`, target: 'banner' }),
  TOO_MANY_ACTIVE: (m) => ({ msg: `You have ${m.activeCount || ''} chapters processing. Wait for one to finish before submitting more.`, target: 'banner' }),
  BUDGET_REACHED: () => ({ msg: "Today's processing budget has run out. Resets at midnight UTC. Sorry!", target: 'banner' }),
  TURNSTILE_FAILED: () => ({ msg: "The bot check didn't go through. Please try submitting again.", target: 'banner', sideEffect: 'reset-turnstile' }),
  LLM_TIMEOUT: () => ({ msg: "Translation server is taking too long. Try this chapter again in a minute.", target: 'job' }),
  OCR_TIMEOUT: () => ({ msg: "Text recognition is slow. Try this chapter again in a minute.", target: 'job' }),
  INTERRUPTED: () => ({ msg: "The server restarted while this chapter was processing. Please resubmit.", target: 'job' }),
  INTERNAL: () => ({ msg: "Something went wrong. Try again, or wait a minute if it keeps happening.", target: 'banner' }),
};

function formatRetryAfter(seconds) {
  if (!seconds || seconds <= 0) return 'a moment';
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  const hours = Math.ceil(seconds / 3600);
  return hours === 1 ? '1 hour' : `${hours}h`;
}

function showError(code, meta = {}) {
  const builder = ERRORS[code] || ERRORS.INTERNAL;
  const { msg, target, sideEffect } = builder(meta);
  if (sideEffect === 'reset-turnstile') resetTurnstile();
  if (target === 'job') return msg;
  showErrorMsg(msg);
  return msg;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function relativeTime(ms) {
  const sec = Math.floor((Date.now() - ms) / 1000);
  const min = Math.floor(sec / 60), hr = Math.floor(min / 60), day = Math.floor(hr / 24);
  if (sec < 60) return 'just now';
  if (min < 60) return `${min}m ago`;
  if (hr < 24) return `${hr}h ago`;
  if (day === 1) return 'yesterday';
  if (day < 7) return `${day}d ago`;
  return `${Math.floor(day / 7)}w ago`;
}

// ---- State ----
let queue = [];
let pollInterval = null;
let jobHistory = [];
let inputLanguage = 'auto';

// ---- DOM ----
const form = document.getElementById('process-form');
const urlRows = document.getElementById('url-rows');
const addUrlBtn = document.getElementById('add-url-btn');
const duplicateMsg = document.getElementById('url-duplicate-msg');
// Max chapter URLs per submission — matches the server's per-IP ACTIVE_JOB_CAP.
// At this many rows the "Add another URL" button is hidden rather than letting
// the user add a 4th and hit a rejection on submit.
const MAX_URLS = 3;
const langSegmented = document.getElementById('lang-segmented');
const modeHint = document.getElementById('mode-hint');
const submitBtn = document.getElementById('submit-btn');
const submitHint = document.getElementById('submit-hint');
const errorMsg = document.getElementById('error-msg');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const lowLevelWarning = document.getElementById('low-level-warning');

const views = { start: document.getElementById('view-start'), run: document.getElementById('view-run'), flashcards: document.getElementById('view-flashcards') };
const runJobsEl = document.getElementById('run-jobs');
const runTitle = document.getElementById('run-title');
const runEta = document.getElementById('run-eta');
const runBack = document.getElementById('run-back');
const newBatchBtn = document.getElementById('new-batch-btn');

const brand = document.getElementById('brand');
const recentBtn = document.getElementById('recent-btn');
const flashcardsBtn = document.getElementById('flashcards-btn');
const drawerBackdrop = document.getElementById('drawer-backdrop');
const recentDrawer = document.getElementById('recent-drawer');
const drawerClose = document.getElementById('drawer-close');
const recentList = document.getElementById('recent-list');

// ---- Icons ----
const ICON = {
  download: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3',
  analyze: 'M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2M7 8h10M7 12h10M7 16h6',
  simplify: 'm12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z',
  build: 'M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z',
  check: 'M20 6 9 17l-5-5',
  book: 'M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z',
  arrow: 'M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6',
  chevron: 'm9 18 6-6-6-6',
};
function svgIcon(path) { return `<svg class="icon" viewBox="0 0 24 24"><path d="${path}"/></svg>`; }

const STEPS = [
  { key: 'download', name: 'Downloading chapter', sub: 'Fetching every page image from the source site', icon: ICON.download },
  { key: 'analyze', name: 'Reading & analyzing text', sub: 'Recognizing the text, then a context pass for names and recurring terms', icon: ICON.analyze },
  { key: 'simplify', name: 'Simplifying & cleaning pages', sub: 'Rewriting the dialogue to your level, then redrawing it onto each page', icon: ICON.simplify },
  { key: 'build', name: 'Building your reader', sub: 'Assembling the reader with pinyin and an original/simplified toggle', icon: ICON.build },
];

// ============ View switching ============
function showView(name) {
  Object.entries(views).forEach(([k, el]) => el.classList.toggle('active', k === name));
  window.scrollTo(0, 0);
}

// Brand is a native <a href="/"> link — navigates to the landing page. The
// old in-app view-switcher click handler was removed when the route migration
// made / the marketing landing (separate from the form at /start).
flashcardsBtn.addEventListener('click', () => { showView('flashcards'); renderFlashcards(); });

// ============ Recent drawer ============
function toggleDrawer() {
  const open = recentDrawer.classList.toggle('open');
  drawerBackdrop.classList.toggle('open', open);
  if (open) renderHistory();
}
recentBtn.addEventListener('click', toggleDrawer);
drawerClose.addEventListener('click', toggleDrawer);
drawerBackdrop.addEventListener('click', toggleDrawer);
// Open the drawer programmatically (e.g. right after sign-in) so the claim
// banner + chapter list is surfaced instead of hidden behind the button.
function openRecentDrawer() {
  if (!recentDrawer.classList.contains('open')) toggleDrawer();
  else renderHistory(); // already open → refresh so it reflects the current auth state
}

// ============ Overflow "•••" menu (mobile) ============
// Collapses secondary header items (Flashcards / About / Feedback) behind a
// button on phones so the header stops overflowing; auth + theme stay in the bar.
const navMoreBtn = document.getElementById('nav-more-btn');
const headerMore = document.getElementById('header-more');
if (navMoreBtn && headerMore) {
  navMoreBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = headerMore.classList.toggle('open');
    navMoreBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  headerMore.addEventListener('click', (e) => {
    if (e.target.closest('a, button')) {
      headerMore.classList.remove('open');
      navMoreBtn.setAttribute('aria-expanded', 'false');
    }
  });
  document.addEventListener('click', (e) => {
    if (!headerMore.contains(e.target) && !navMoreBtn.contains(e.target)) {
      headerMore.classList.remove('open');
      navMoreBtn.setAttribute('aria-expanded', 'false');
    }
  });
}

// ============ Theme ============
const MOON = '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>';
const SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>';
function applyTheme(dark) {
  document.body.classList.toggle('dark', dark);
  themeIcon.innerHTML = dark ? SUN : MOON;
}
const savedTheme = localStorage.getItem('theme');
applyTheme(savedTheme === 'dark' || (savedTheme === null && window.matchMedia('(prefers-color-scheme: dark)').matches));
themeToggle.addEventListener('click', () => {
  const dark = !document.body.classList.contains('dark');
  applyTheme(dark);
  localStorage.setItem('theme', dark ? 'dark' : 'light');
});

// ============ Input language (sliding segmented) ============
langSegmented.querySelectorAll('.seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    langSegmented.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    langSegmented.querySelector('.seg-indicator').style.transform = `translateX(${parseInt(btn.dataset.i, 10) * 100}%)`;
    inputLanguage = btn.dataset.lang;
    modeHint.textContent = inputLanguage === 'english'
      ? 'Translate English manga into HSK-level Chinese.'
      : inputLanguage === 'japanese'
      ? 'Translate Japanese webtoons into HSK-level Chinese. Vertical text (traditional paged manga) is not supported yet.'
      : inputLanguage === 'korean'
      ? 'Translate Korean webtoons into HSK-level Chinese.'
      : inputLanguage === 'chinese'
      ? 'Simplify Chinese manga to your HSK level.'
      : 'Language will be detected automatically.';
  });
});

// ============ HSK level ============
const HSK_LEVEL_KEY = 'lastHskLevel';
const savedHsk = localStorage.getItem(HSK_LEVEL_KEY);
if (savedHsk) {
  const r = document.querySelector(`input[name="level"][value="${savedHsk}"]`);
  if (r) r.checked = true;
}
function refreshLowLevelWarning() {
  const checked = document.querySelector('input[name="level"]:checked');
  const isLow = checked && (checked.value === 'hsk1' || checked.value === 'hsk2');
  lowLevelWarning.classList.toggle('hidden', !isLow);
}
refreshLowLevelWarning();
document.querySelectorAll('input[name="level"]').forEach(radio => {
  radio.addEventListener('change', () => {
    if (radio.checked) localStorage.setItem(HSK_LEVEL_KEY, radio.value);
    refreshLowLevelWarning();
  });
});

// ============ URL rows ============
let duplicateTimeout = null;
const URL_REGEX = /^https?:\/\/[^\s]+\.[^\s]+/;

function createUrlRow(value) {
  const wrapper = document.createElement('div');
  wrapper.className = 'url-row-wrapper';
  const row = document.createElement('div');
  row.className = 'url-row';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Paste a chapter URL…';
  input.autocomplete = 'off';
  if (value) { input.value = value; input.classList.add('has-value'); }
  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'url-row-remove';
  removeBtn.textContent = '×';
  removeBtn.setAttribute('aria-label', 'Remove URL');
  const errorMsgEl = document.createElement('div');
  errorMsgEl.className = 'url-error-msg hidden';
  errorMsgEl.textContent = "This doesn't look like a URL";

  function applyValidationStyles() {
    const v = input.value.trim();
    input.classList.toggle('has-value', v.length > 0 && URL_REGEX.test(v));
    const invalid = v.length > 0 && !URL_REGEX.test(v);
    input.classList.toggle('invalid-url', invalid);
    errorMsgEl.classList.toggle('hidden', !invalid);
  }
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault(); e.stopPropagation();
      const val = input.value.trim();
      if (val && !checkDuplicate(val, input)) { applyValidationStyles(); addNewUrlRow(); }
    }
  });
  input.addEventListener('input', applyValidationStyles);
  input.addEventListener('paste', () => setTimeout(() => {
    const val = input.value.trim();
    if (val && checkDuplicate(val, input)) {
      input.value = ''; input.classList.remove('has-value', 'invalid-url'); errorMsgEl.classList.add('hidden');
      return;
    }
    applyValidationStyles();
  }, 0));
  input.addEventListener('change', () => {
    const val = input.value.trim();
    if (val && checkDuplicate(val, input)) return;
    applyValidationStyles();
  });
  removeBtn.addEventListener('click', () => { wrapper.remove(); updateRemoveButtons(); hideDuplicateMsg(); });

  row.appendChild(input);
  row.appendChild(removeBtn);
  wrapper.appendChild(row);
  wrapper.appendChild(errorMsgEl);
  return wrapper;
}
function addNewUrlRow() {
  // Guard: never exceed MAX_URLS rows (the button is hidden at the cap, but
  // this also blocks any other path into here).
  if (urlRows.querySelectorAll('.url-row-wrapper').length >= MAX_URLS) return;
  const w = createUrlRow('');
  urlRows.appendChild(w);
  updateRemoveButtons();
  w.querySelector('input').focus();
}
function updateRemoveButtons() {
  const wrappers = urlRows.querySelectorAll('.url-row-wrapper');
  wrappers.forEach(w => w.querySelector('.url-row-remove').classList.toggle('invisible', wrappers.length <= 1));
  // Hide "Add another URL" once the cap is reached; it reappears if a row is removed.
  addUrlBtn.classList.toggle('hidden', wrappers.length >= MAX_URLS);
}
function checkDuplicate(url, currentInput) {
  for (const inp of urlRows.querySelectorAll('.url-row input')) {
    if (inp !== currentInput && inp.value.trim() === url) { showDuplicateMsg(); return true; }
  }
  hideDuplicateMsg();
  return false;
}
function showDuplicateMsg() {
  duplicateMsg.classList.remove('hidden');
  if (duplicateTimeout) clearTimeout(duplicateTimeout);
  duplicateTimeout = setTimeout(hideDuplicateMsg, 3000);
}
function hideDuplicateMsg() { duplicateMsg.classList.add('hidden'); }
function getUrlsFromRows() {
  const urls = [];
  urlRows.querySelectorAll('.url-row input').forEach(i => { const v = i.value.trim(); if (v) urls.push(v); });
  return urls;
}
function initUrlRows(values) {
  urlRows.innerHTML = '';
  const vals = (values && values.length) ? values : [''];
  vals.forEach(v => urlRows.appendChild(createUrlRow(v)));
  updateRemoveButtons();
}

// Append chapter URLs to the URL box. Each Add click is cumulative — adds
// new rows rather than replacing previous ones, so users can build up a
// list of chapters across multiple Add clicks (including across different
// search results). The first empty placeholder row is reused; further URLs
// open new rows up to MAX_URLS. Duplicates are skipped silently.
// Returns { added, dropped, duplicates } so the caller can show a useful
// status line.
function appendChapterUrlsToBox(urls) {
  let added = 0, dropped = 0, duplicates = 0;
  for (const url of urls) {
    // Skip if this URL is already in the box (re-clicking Add with the same
    // chapter shouldn't duplicate the row).
    const existing = Array.from(urlRows.querySelectorAll('.url-row input'));
    if (existing.some(i => i.value.trim() === url)) { duplicates++; continue; }

    // Try to reuse the first empty placeholder row if there is one.
    const placeholder = existing.find(i => !i.value.trim());
    if (placeholder) {
      placeholder.value = url;
      placeholder.dispatchEvent(new Event('input', { bubbles: true }));
      added++;
      continue;
    }

    // Otherwise add a new row, respecting the MAX_URLS cap.
    if (urlRows.querySelectorAll('.url-row-wrapper').length >= MAX_URLS) {
      dropped++;
      continue;
    }
    const w = createUrlRow(url);
    urlRows.appendChild(w);
    added++;
  }
  updateRemoveButtons();
  return { added, dropped, duplicates };
}

// Parse the chapter textbox value into one of:
//   { type: 'single', list: ['5'] }
//   { type: 'range',  list: ['5','6','7','8','9','10'] }
//   { type: 'invalid' }
// Range accepts ASCII hyphen, en-dash, em-dash. Decimals are kept as-is
// for single inputs (sub-chapters like "5.5") but ranges must be integer.
function parseChapterRange(value) {
  const t = String(value || '').trim();
  if (!t) return { type: 'invalid' };
  if (/^\d+(?:\.\d+)?$/.test(t)) return { type: 'single', list: [t] };
  const m = t.match(/^(\d+)\s*[-–—]\s*(\d+)$/);
  if (!m) return { type: 'invalid' };
  const start = parseInt(m[1], 10);
  const end = parseInt(m[2], 10);
  if (!isFinite(start) || !isFinite(end) || start > end) return { type: 'invalid' };
  const list = [];
  for (let i = start; i <= end; i++) list.push(String(i));
  return { type: 'range', list };
}
addUrlBtn.addEventListener('click', addNewUrlRow);

// ============ Form submission ============
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const urls = getUrlsFromRows();
  const selectedRadio = document.querySelector('input[name="level"]:checked');
  const selectedLevel = selectedRadio ? selectedRadio.value : null;

  if (!selectedLevel) { showErrorMsg('Please select an HSK level.'); return; }
  if (urls.length === 0) { showErrorMsg('Please add at least one chapter URL.'); return; }
  for (const url of urls) {
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      showErrorMsg('All URLs must start with http:// or https://'); return;
    }
  }
  if (turnstileSiteKey && !turnstileToken) { showErrorMsg('Please complete the bot check before submitting.'); return; }

  submitBtn.disabled = true;
  submitBtn.querySelector('svg')?.remove();
  submitBtn.textContent = 'Starting…';
  hideErrorMsg();

  try {
    const response = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls, levels: [selectedLevel], includeEnglish: false, inputLanguage, turnstileToken }),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.code || 'INTERNAL', data);
      restoreSubmitBtn();
      resetTurnstile();
      return;
    }
    resetTurnstile();
    queue = data.jobs.map((job, i) => ({
      jobId: job.jobId,
      url: urls[i],
      title: null,
      status: 'pending',
      message: job.fromCache ? 'Queued (cache hit)…' : 'Queued…',
      resultUrl: null,
      level: selectedLevel,
      lang: inputLanguage,
      errorCode: null,
      errorMsg: null,
      fromCache: !!job.fromCache,
    }));
    saveQueue();
    enterRunView();
    startPolling();
  } catch (err) {
    showErrorMsg('Network error: ' + err.message);
    restoreSubmitBtn();
  }
});

function restoreSubmitBtn() {
  submitBtn.disabled = false;
  submitBtn.innerHTML = `${svgIcon(ICON.simplify)}Simplify chapter`;
}

// ============ Run view ============
function enterRunView() {
  showView('run');
  newBatchBtn.classList.add('hidden');
  renderRunJobs();
  syncRunJobs();
}

function renderRunJobs() {
  runJobsEl.innerHTML = '';
  renderQueueAhead(0);
  queue.forEach(item => {
    const card = document.createElement('div');
    card.className = 'job-card';
    card.dataset.job = item.jobId;
    card.innerHTML = `
      <div class="job-card-head">
        <span class="job-card-title">${escapeHtml(item.title || shortenUrl(item.url))}</span>
        <span class="job-chip">Queued</span>
      </div>
      <div class="stepper hidden">
        ${STEPS.map((s, i) => `
          <div class="step pending" data-step="${i}">
            <div class="step-node">${svgIcon(s.icon)}</div>
            <div class="step-body">
              <div class="step-name">${s.name}</div>
              <div class="step-sub">${s.sub}</div>
            </div>
          </div>`).join('')}
      </div>
      <div class="job-done hidden">
        <div class="job-done-badge">${svgIcon(ICON.check)}</div>
        <h3>Your chapter is ready</h3>
        <p class="job-done-name"></p>
        <a class="job-open" target="_blank" href="#">${svgIcon(ICON.book)}Open reader</a>
      </div>
      <div class="job-error-wrap hidden">
        <div class="job-error-box"></div>
        <button class="job-retry" type="button">Try again</button>
      </div>`;
    card.querySelector('.job-retry').addEventListener('click', () => retryJob(item.url));
    runJobsEl.appendChild(card);
  });
}

function currentStepFor(item) {
  const msg = (item.message || '').toLowerCase();
  if (item.status === 'complete') return 4;
  if (item.status === 'downloading') return 0;
  if (item.status === 'processing') {
    // Keywords below match the pipeline's actual progress messages. The
    // production "batched" path emits "Scanning pages…" (OCR) then "Rendering
    // pages…" (Phase 4b: LLM + inpaint); the non-batched path emits
    // "Simplifying and cleaning pages…". "Building reader…" ends both.
    if (msg.includes('reading') || msg.includes('analyzing') || msg.includes('scanning')) return 1;
    if (msg.includes('simplifying') || msg.includes('cleaning') || msg.includes('rendering')) return 2;
    if (msg.includes('building') || msg.includes('reader')) return 3;
    return 1;
  }
  return -1; // pending
}

function syncRunJobs() {
  let done = 0, errored = 0;
  queue.forEach(item => {
    const card = runJobsEl.querySelector(`.job-card[data-job="${CSS.escape(item.jobId)}"]`);
    if (!card) return;
    card.querySelector('.job-card-title').textContent = item.title || shortenUrl(item.url);
    const chip = card.querySelector('.job-chip');
    const stepper = card.querySelector('.stepper');
    const doneBlock = card.querySelector('.job-done');
    const errBlock = card.querySelector('.job-error-wrap');
    card.classList.toggle('queued', item.status === 'pending');

    if (item.status === 'error') {
      errored++;
      chip.textContent = 'Failed'; chip.className = 'job-chip error';
      stepper.classList.add('hidden'); doneBlock.classList.add('hidden');
      errBlock.classList.remove('hidden');
      errBlock.querySelector('.job-error-box').textContent = item.errorMsg || item.error || 'Something went wrong.';
    } else if (item.status === 'complete') {
      done++;
      chip.innerHTML = `${svgIcon(ICON.check)}Done`; chip.className = 'job-chip done';
      stepper.classList.add('hidden'); errBlock.classList.add('hidden');
      doneBlock.classList.remove('hidden');
      doneBlock.querySelector('.job-done-name').textContent = item.title || '';
      if (item.resultUrl) doneBlock.querySelector('.job-open').href = item.resultUrl;
    } else {
      const isPending = item.status === 'pending';
      // Queued cards show their real position ("Next up" / "#3 in queue") once
      // the server has reported it; fall back to a plain "Queued" pre-poll.
      let pendingLabel = 'Queued';
      if (isPending && typeof item.jobsAhead === 'number') {
        pendingLabel = item.jobsAhead === 0 ? 'Next up' : `#${item.jobsAhead + 1} in queue`;
      }
      chip.textContent = isPending ? pendingLabel : 'Processing';
      chip.className = 'job-chip';
      // Queued chapters stay slim — the stepper appears only once the job runs.
      stepper.classList.toggle('hidden', isPending);

      const cur = currentStepFor(item);
      card.querySelectorAll('.step').forEach((stepEl, i) => {
        stepEl.classList.remove('done', 'active', 'pending');
        const node = stepEl.querySelector('.step-node');
        if (i < cur) { stepEl.classList.add('done'); node.innerHTML = svgIcon(ICON.check); }
        else if (i === cur) { stepEl.classList.add('active'); node.innerHTML = '<div class="spinner"></div>'; }
        else { stepEl.classList.add('pending'); node.innerHTML = svgIcon(STEPS[i].icon); }
      });
    }
  });

  const total = queue.length;
  if (done + errored === total) {
    if (errored === total) runTitle.textContent = total === 1 ? "That chapter didn't go through" : 'Those chapters didn\'t go through';
    else if (errored > 0) runTitle.textContent = `${done} of ${total} chapters ready`;
    else runTitle.textContent = total === 1 ? 'Your chapter is ready' : `${total} chapters ready`;
    runEta.classList.add('hidden');
    newBatchBtn.classList.remove('hidden');
  } else {
    runTitle.textContent = total === 1 ? 'Processing your chapter…' : `Processing ${total} chapters…`;
    runEta.classList.remove('hidden');
  }
}

// One box above the job cards: chapters from OTHER users ahead of this batch.
// Hidden when 0 (nobody else ahead, or all of this user's jobs have started).
function renderQueueAhead(n) {
  const box = document.getElementById('queue-ahead');
  if (!box) return;
  if (typeof n === 'number' && n > 0) {
    document.getElementById('queue-ahead-title').textContent =
      `${n} chapter${n === 1 ? '' : 's'} ahead of you`;
    box.classList.remove('hidden');
  } else {
    box.classList.add('hidden');
  }
}

function retryJob(url) {
  resetToStart();
  initUrlRows([url]);
}

runBack.addEventListener('click', resetToStart);
newBatchBtn.addEventListener('click', resetToStart);

function resetToStart() {
  stopPolling();
  queue = [];
  localStorage.removeItem('processingQueue');
  initUrlRows();
  restoreSubmitBtn();
  hideErrorMsg();
  runEta.classList.remove('hidden');
  newBatchBtn.classList.add('hidden');
  showView('start');
}

// ============ Queue persistence + polling ============
function saveQueue() { localStorage.setItem('processingQueue', JSON.stringify(queue)); }

function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(checkAllStatuses, 2000);
  checkAllStatuses();
}
function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

async function checkAllStatuses() {
  let allDone = true;
  let chaptersAhead = 0;
  for (const item of queue) {
    if (item.status === 'complete' || item.status === 'error') continue;
    allDone = false;
    try {
      const response = await fetch(`/api/status/${item.jobId}`);
      if (!response.ok) {
        item.status = 'error';
        item.errorCode = 'INTERNAL';
        item.errorMsg = 'This job could not be found. Please resubmit.';
        continue;
      }
      const data = await response.json();
      item.status = data.status;
      item.message = data.message;
      item.jobsAhead = typeof data.jobsAhead === 'number' ? data.jobsAhead : undefined;
      if (typeof data.chaptersAhead === 'number') chaptersAhead = data.chaptersAhead;
      if (data.title) item.title = data.title;
      if (data.resolved) item.resolved = data.resolved;
      if (data.resultUrl) item.resultUrl = data.resultUrl;
      if (data.error) item.error = data.error;
      if (data.status === 'error') {
        const code = data.errorCode || 'INTERNAL';
        item.errorCode = code;
        item.errorMsg = (ERRORS[code] || ERRORS.INTERNAL)(data).msg;
      }
    } catch (err) {
      console.error('Poll error for job', item.jobId, err);
    }
  }
  syncRunJobs();
  renderQueueAhead(allDone ? 0 : chaptersAhead);
  saveQueue();
  if (allDone) {
    stopPolling();
    onAllJobsDone();
    localStorage.removeItem('processingQueue');
  }
}

let authReady = null; // resolves when /api/me has been checked (see fetchCurrentUser call)
async function onAllJobsDone() {
  // Wait for the auth check so recordChapter routes a signed-in user's chapter to
  // the account, not the browser bucket (avoids a spurious "unclaimed" item).
  if (authReady) { try { await authReady; } catch (e) {} }
  refreshEta();
  queue.forEach(item => {
    if (item.status === 'complete' && item.resultUrl) {
      recordChapter(item.url, item.resultUrl, item.title, item.level, item.lang, item.resolved);
    }
  });
}

// ============ Error banner ============
function showErrorMsg(message) { errorMsg.textContent = message; errorMsg.classList.remove('hidden'); }
function hideErrorMsg() { errorMsg.classList.add('hidden'); }

// ============ My chapters (drawer) ============
// Two buckets:
//  - Anonymous device bucket: localStorage `jobHistory` — chapters processed
//    while signed OUT. Each entry keeps its resultUrl (openable). Uncapped.
//  - Account library: server-side (GET /api/library), shown when signed in.
// Signed-in processing goes STRAIGHT to the account; the anonymous bucket is
// pulled in only by an explicit claim (the drawer's "add to account" banner).
// This separation is what prevents one account's chapters from leaking into
// another account signed in on the same browser.
const ANON_CAP = 500; // entries are ~100 bytes; this is a runaway guard, not a UX cap
function shortenUrl(url) { return url.length > 46 ? url.slice(0, 43) + '…' : url; }
function hskLabel(level) { const n = String(level || '').replace(/\D/g, ''); return n ? 'HSK ' + n : ''; }

function loadHistory() {
  try { jobHistory = JSON.parse(localStorage.getItem('jobHistory') || '[]'); }
  catch (e) { jobHistory = []; }
}
function saveAnon() { localStorage.setItem('jobHistory', JSON.stringify(jobHistory)); }

// Record a finished chapter. Signed in → account library (with resultUrl, so
// other devices open it directly); signed out → anonymous device bucket.
function recordChapter(url, resultUrl, title, level, lang, resolved) {
  if (currentUser) {
    fetch('/api/library/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ url, resultUrl, title: title || '', level: level || 'hsk3', lang: lang || 'auto', resolved: resolved || undefined }),
    }).catch(() => {});
  } else {
    const lvl = level || 'hsk3';
    const i = jobHistory.findIndex(h => h.url === url && (h.level || 'hsk3') === lvl);
    if (i >= 0) jobHistory.splice(i, 1);
    jobHistory.unshift({ url, resultUrl, title: title || null, level: lvl, lang: lang || null, resolved: resolved || undefined, timestamp: Date.now() });
    if (jobHistory.length > ANON_CAP) jobHistory = jobHistory.slice(0, ANON_CAP);
    saveAnon();
  }
}

function chapterRowHtml(item) {
  const hasReader = !!item.resultUrl;
  const href = hasReader ? escapeHtml(item.resultUrl) : ('/start?url=' + encodeURIComponent(item.url));
  const target = hasReader ? '_blank' : '_self';
  const subLabel = hasReader ? relativeTime(item.timestamp || Date.now()) : 'Click to load';
  const badge = item.level ? `<span class="recent-level">${escapeHtml(hskLabel(item.level))}</span>` : '';

  // Learner-friendly resolution, when present. Title = English, else the Chinese
  // series name; never a fabricated name. A cover replaces the book icon; a
  // hanzi + pinyin aid line and the normalized chapter number ride below.
  const r = item.resolved;
  const titleText = (r && (r.english || r.seriesZh)) || item.title || shortenUrl(item.url);
  const thumb = (r && r.coverUrl)
    ? `<span class="recent-cover"><img src="${escapeHtml(r.coverUrl)}" alt="" loading="lazy"></span>`
    : `<span class="recent-thumb">${svgIcon(ICON.book)}</span>`;
  // The Chinese + pinyin aid line only applies to actual Han series names; an
  // English title never gets one (its "pinyin" would just echo the English).
  const hasHan = !!(r && r.seriesZh && /[一-鿿]/.test(r.seriesZh));
  let cnLine = '';
  if (hasHan && r.english) {
    cnLine = `<span class="recent-cn">${escapeHtml(r.seriesZh)}${r.seriesPinyin ? ` · <span class="recent-py">${escapeHtml(r.seriesPinyin)}</span>` : ''}</span>`;
  } else if (hasHan && r.seriesPinyin) {
    cnLine = `<span class="recent-cn"><span class="recent-py">${escapeHtml(r.seriesPinyin)}</span></span>`;
  }
  const chap = (r && r.chapterLabel) ? `<span class="recent-chap">${escapeHtml(r.chapterLabel)}</span>` : '';

  return `
    <div class="recent-item-wrap" data-url="${escapeHtml(item.url)}" data-level="${escapeHtml(item.level || '')}" style="display:flex;align-items:center;gap:6px">
      <a class="recent-item" href="${href}" target="${target}" style="flex:1">
        ${thumb}
        <span class="recent-meta">
          <span class="recent-title">${escapeHtml(titleText)}</span>
          ${cnLine}
          <span class="recent-sub">${badge}${chap}${escapeHtml(subLabel)}</span>
        </span>
        <span class="recent-thumb" style="background:none;color:var(--muted);width:auto;height:auto">${svgIcon(ICON.chevron)}</span>
      </a>
      <button class="recent-del" type="button" aria-label="Remove">×</button>
    </div>`;
}

// Async. Signed in → the account library PLUS any not-yet-claimed browser
// chapters (so you can SEE what the claim banner is offering to sync, not just a
// count). Signed out → the browser bucket.
async function renderHistory() {
  let accountItems = [];
  let libraryOk = !currentUser; // signed out → nothing to fetch
  if (currentUser) {
    try {
      const r = await fetch('/api/library', { credentials: 'same-origin' });
      const body = await r.json().catch(() => ({}));
      if (body && body.signedIn === false) {
        // Session expired server-side — reconcile client auth and render as guest.
        currentUser = null;
        updateAuthPill();
        libraryOk = true;
      } else if (body && body.signedIn && Array.isArray(body.library)) {
        accountItems = body.library.map(e => ({ url: e.url, resultUrl: e.resultUrl, title: e.title, resolved: e.resolved, level: e.level, timestamp: e.addedAt }));
        libraryOk = true;
      }
    } catch (e) {}
  }
  // Signed in but the library didn't load: don't blank the account or mislabel
  // bucket chapters as "unclaimed" — offer a retry instead.
  if (currentUser && !libraryOk) {
    recentList.innerHTML = '<p class="drawer-empty">Couldn’t load your chapters.<br><button type="button" id="library-retry" style="margin-top:10px;background:none;border:none;color:var(--accent);font:inherit;font-weight:700;cursor:pointer;text-decoration:underline">Retry</button></p>';
    const rb = document.getElementById('library-retry');
    if (rb) rb.addEventListener('click', renderHistory);
    return;
  }
  // Browser chapters not already in the account (deduped by url+level, so the
  // same chapter at a different HSK level still shows).
  const accKeys = new Set(accountItems.map(i => i.url + '|' + (i.level || 'hsk3')));
  const anonItems = (currentUser ? jobHistory.filter(h => !accKeys.has(h.url + '|' + (h.level || 'hsk3'))) : jobHistory)
    .map(h => ({ url: h.url, resultUrl: h.resultUrl, title: h.title, resolved: h.resolved, level: h.level, timestamp: h.timestamp }));
  // Only http URLs can be synced to an account (image uploads carry a non-http
  // placeholder url and can't be claimed) — count/gate the banner on those, but
  // still SHOW upload rows so they're visible/deletable.
  const claimableCount = anonItems.filter(h => typeof h.url === 'string' && h.url.startsWith('http')).length;
  const showClaim = currentUser && claimableCount > 0;
  // Unclaimed browser chapters first (right under the banner that references
  // them), then the already-synced account chapters.
  const items = currentUser ? anonItems.concat(accountItems) : anonItems;

  let bannerHtml = '';
  let statusHtml = '';
  if (currentUser) {
    if (showClaim) {
      const n = claimableCount;
      bannerHtml = `<div class="claim-banner"><span>${n} chapter${n === 1 ? '' : 's'} in your browser ${n === 1 ? "isn't" : "aren't"} in your account yet.</span><button type="button" id="claim-btn">Add to my account</button></div>`;
    } else if (items.length) {
      statusHtml = `<p class="drawer-status">Synced to your account · ${escapeHtml(currentUser.email || '')}</p>`;
    }
  } else if (items.length) {
    statusHtml = `<p class="drawer-status">Saved to your browser. Sign in to read your chapters on any device.</p>`;
  }
  if (!items.length && !showClaim) {
    // Signed out + empty: reassure that chapters may live in an account (not lost).
    const hint = currentUser ? '' : '<span style="display:block;margin-top:10px;font-size:12px;font-weight:600;opacity:.8">Sign in to see any chapters saved to your account.</span>';
    recentList.innerHTML = '<p class="drawer-empty">No chapters yet' + hint + '</p>';
    return;
  }
  recentList.innerHTML = bannerHtml + statusHtml + items.map(chapterRowHtml).join('');
  const cb = document.getElementById('claim-btn');
  if (cb) cb.addEventListener('click', claimAnonymousChapters);
}

// Explicit claim: move the anonymous device bucket into the signed-in account,
// then clear the bucket. The ONLY path by which pre-existing chapters enter an
// account.
async function claimAnonymousChapters() {
  const entries = jobHistory
    .filter(h => h && typeof h.url === 'string' && h.url.startsWith('http'))
    .slice(0, 500)
    .map(h => ({ url: h.url, resultUrl: h.resultUrl || undefined, title: h.title || '', resolved: h.resolved || undefined, level: h.level || 'hsk3', lang: h.lang || 'auto' }));
  if (!entries.length) return;
  try {
    const r = await fetch('/api/library/claim', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ entries }),
    });
    if (r.ok) {
      const body = await r.json().catch(() => ({}));
      const added = typeof body.added === 'number' ? body.added : entries.length;
      // Clear only the entries we actually sent (preserve non-http/upload chapters
      // the http filter excluded), keyed the same way (url|level) they were built.
      const claimedKeys = new Set(entries.map(e => e.url + '|' + e.level));
      jobHistory = jobHistory.filter(h => !claimedKeys.has(h.url + '|' + (h.level || 'hsk3')));
      saveAnon();
      renderHistory();
      if (typeof showToast === 'function') showToast(`Added ${added} chapter${added === 1 ? '' : 's'} to your account`);
    }
  } catch (e) {}
}
// Opening a chapter from the list → stamp lastReadAt on the account library
// entry (fire-and-forget; server no-ops for guests). This is what makes
// "chapters read over time" measurable per account.
recentList.addEventListener('click', (e) => {
  const link = e.target.closest('.recent-item');
  if (!link || !currentUser) return;
  const wrap = link.closest('.recent-item-wrap');
  if (!wrap || !wrap.dataset.url) return;
  try {
    fetch('/api/library/touch', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: wrap.dataset.url, level: wrap.dataset.level || undefined }),
      keepalive: true,
    });
  } catch (err) {}
});
recentList.addEventListener('click', (e) => {
  const del = e.target.closest('.recent-del');
  if (!del) return;
  e.preventDefault();
  const wrap = del.closest('.recent-item-wrap');
  const url = wrap && wrap.dataset.url;
  if (!url) return;
  const level = (wrap && wrap.dataset.level) || undefined;
  if (!confirm('Remove this chapter from your list?')) return;
  // Route by ROW, not auth state: one displayed row can be backed by the bucket,
  // the account, or both (after url+level dedup). Always drop the bucket copy;
  // only hit the account endpoint with a CONCRETE level — a blank level means a
  // bucket-only row, and an omitted level would delete every level of that url.
  jobHistory = jobHistory.filter(h => !(h.url === url && (h.level || 'hsk3') === (level || 'hsk3')));
  saveAnon();
  if (currentUser && level) {
    fetch('/api/library/remove', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ url, level }),
    }).catch(() => {}).then(() => renderHistory());
  } else {
    renderHistory();
  }
});

// ============ Turnstile ============
let turnstileWidgetId = null;
let turnstileToken = null;
let turnstileSiteKey = null;

async function initTurnstile() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();
    turnstileSiteKey = cfg.turnstileSiteKey;
    if (!turnstileSiteKey) return;
    const script = document.createElement('script');
    script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
    script.async = true; script.defer = true;
    script.onload = () => {
      const mount = document.getElementById('turnstile-mount');
      if (!mount || !window.turnstile) return;
      mount.classList.remove('hidden');
      turnstileWidgetId = window.turnstile.render(mount, {
        sitekey: turnstileSiteKey,
        callback: (t) => { turnstileToken = t; },
        'expired-callback': () => { turnstileToken = null; },
        'error-callback': () => { turnstileToken = null; },
      });
    };
    document.head.appendChild(script);
  } catch (e) { console.warn('[Turnstile] init failed:', e); }
}
function resetTurnstile() {
  if (turnstileWidgetId !== null && window.turnstile) {
    try { window.turnstile.reset(turnstileWidgetId); } catch (e) {}
  }
  turnstileToken = null;
}

// ============ ETA ============
async function refreshEta() {
  try {
    const res = await fetch('/api/eta');
    if (!res.ok) return;
    const data = await res.json();
    const txt = (data.cold && data.cold.median)
      ? `Estimated ~${formatEtaSeconds(Math.round(data.cold.median / 1000))} per chapter.`
      : (data.fallback || 'Usually under 2 minutes per chapter.');
    if (submitHint) submitHint.textContent = txt;
    if (runEta) runEta.textContent = txt;
  } catch (e) { /* keep existing */ }
}
function formatEtaSeconds(s) {
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60), r = s % 60;
  return r === 0 ? `${m} min` : `${m} min ${r}s`;
}

// ============ Page load ============
window.addEventListener('DOMContentLoaded', () => {
  initUrlRows();
  loadHistory();
  initTurnstile();
  refreshEta();
  renderFlashcards();

  const savedQueue = localStorage.getItem('processingQueue');
  if (savedQueue) {
    try {
      queue = JSON.parse(savedQueue);
      if (queue.length > 0) {
        enterRunView();
        const unfinished = queue.some(j => j.status !== 'complete' && j.status !== 'error');
        if (unfinished) startPolling(); else { syncRunJobs(); onAllJobsDone(); }
      }
    } catch (e) { localStorage.removeItem('processingQueue'); }
  }
});

// ============================================================
// Flashcards
// ============================================================
const FLASHCARDS_KEY = 'flashcards';

function loadFlashcards() {
  try {
    const raw = localStorage.getItem(FLASHCARDS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) { return {}; }
}
function saveFlashcards(map) { try { localStorage.setItem(FLASHCARDS_KEY, JSON.stringify(map)); } catch (e) {} }
function flashcardsAsList(map) { return Object.values(map).sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0)); }

const flashcardsBadge = document.getElementById('flashcards-badge');
const flippedCards = new Set();
let showAll = false;

// Script display preference, shared with the reader's Traditional toggle via the
// same localStorage key so the choice follows the learner across both surfaces.
// Display only: cards remain keyed on Simplified (card.hanzi), with the
// Traditional form carried on card.trad when it differs. Cards saved before this
// feature have no .trad and fall back to .hanzi.
const SCRIPT_KEY = 'hm_script';
let cardScript = 'simp';
try { if (localStorage.getItem(SCRIPT_KEY) === 'trad') cardScript = 'trad'; } catch (e) {}

function cardDisplayHanzi(card) {
  return (cardScript === 'trad' && card.trad) ? card.trad : card.hanzi;
}

function timeAgo(ts) { return relativeTime(ts); }

function applyFlashcardsFilter(cards) {
  const search = (document.getElementById('flashcards-search')?.value || '').toLowerCase().trim();
  const filterHsk = document.getElementById('flashcards-filter-hsk')?.value || 'all';
  return cards.filter(card => {
    if (filterHsk === 'above' && card.hsk <= 4) return false;
    if (filterHsk !== 'all' && filterHsk !== 'above' && card.hsk !== parseInt(filterHsk, 10)) return false;
    // Hanzi search matches EITHER script regardless of display mode — someone
    // typing 发 should still find the card they are viewing as 發, and vice versa.
    const hanziHit = search && (card.hanzi.includes(search) || (card.trad || '').includes(search));
    if (search && !(hanziHit || (card.pinyin || '').toLowerCase().includes(search) || (card.meaning || '').toLowerCase().includes(search))) return false;
    return true;
  });
}

function renderFlashcards() {
  const grid = document.getElementById('flashcards-grid');
  const empty = document.getElementById('flashcards-empty');
  const subtitle = document.getElementById('flashcards-subtitle');
  const toolbar = document.querySelector('.flashcards-toolbar');
  const actions = document.querySelector('.flashcards-actions');
  if (!grid) return;

  const all = flashcardsAsList(loadFlashcards());
  const total = all.length;

  if (total > 0) { flashcardsBadge.textContent = String(total); flashcardsBadge.classList.remove('hidden'); }
  else flashcardsBadge.classList.add('hidden');

  if (total === 0) {
    grid.innerHTML = ''; grid.classList.add('hidden');
    empty.classList.remove('hidden');
    toolbar.classList.add('hidden'); actions.classList.add('hidden');
    subtitle.textContent = 'Save words from chapters to study them here';
    return;
  }
  empty.classList.add('hidden');
  grid.classList.remove('hidden');
  toolbar.classList.remove('hidden'); actions.classList.remove('hidden');
  grid.classList.toggle('show-all', showAll);

  const visible = applyFlashcardsFilter(all);
  subtitle.textContent = total === visible.length
    ? `${total} word${total === 1 ? '' : 's'} saved`
    : `${total} saved · showing ${visible.length}`;

  grid.innerHTML = visible.map(card => {
    const hskClass = (card.hsk || 0) > 4 ? 'hsk-pill above' : 'hsk-pill';
    const flipped = flippedCards.has(card.hanzi) ? ' flipped' : '';
    // `h` is the storage key (always Simplified) and stays in data-hanzi — the
    // remove button and flip tracking both look up by it. `d` is the displayed
    // form. Keeping these separate is what stops the script toggle from
    // orphaning a card's remove/flip behaviour.
    const h = escapeHtml(card.hanzi), p = escapeHtml(card.pinyin || ''), m = escapeHtml(card.meaning || '');
    const d = escapeHtml(cardDisplayHanzi(card));
    const badge = `<span class="${hskClass}">HSK ${card.hsk || '?'}</span>`;
    const saved = `<span class="flashcard-saved">${timeAgo(card.savedAt || Date.now())}</span>`;
    return `
      <div class="flashcard" data-hanzi="${h}">
        <button class="btn-card-remove" data-hanzi="${h}" title="Remove">×</button>
        <div class="flashcard-inner${flipped}">
          <div class="flashcard-face flashcard-front">
            <div class="flashcard-hanzi">${d}</div>
            <div class="flashcard-front-hint">tap to flip</div>
            <div class="flashcard-footer">${badge}${saved}</div>
          </div>
          <div class="flashcard-face flashcard-back">
            <div class="flashcard-hanzi">${d}</div>
            <div class="flashcard-pinyin">${p}</div>
            <div class="flashcard-meaning">${m}</div>
            <div class="flashcard-footer">${badge}${saved}</div>
          </div>
        </div>
      </div>`;
  }).join('');
}

const flashcardsGridEl = document.getElementById('flashcards-grid');
if (flashcardsGridEl) {
  flashcardsGridEl.addEventListener('click', (e) => {
    const removeBtn = e.target.closest('.btn-card-remove');
    if (removeBtn) {
      e.stopPropagation();
      const map = loadFlashcards();
      delete map[removeBtn.dataset.hanzi];
      saveFlashcards(map);
      flippedCards.delete(removeBtn.dataset.hanzi);
      renderFlashcards();
      return;
    }
    if (showAll) return;
    const card = e.target.closest('.flashcard');
    if (!card) return;
    const hanzi = card.dataset.hanzi;
    const inner = card.querySelector('.flashcard-inner');
    if (flippedCards.has(hanzi)) { flippedCards.delete(hanzi); inner.classList.remove('flipped'); }
    else { flippedCards.add(hanzi); inner.classList.add('flipped'); }
  });
}
document.getElementById('flashcards-search')?.addEventListener('input', renderFlashcards);
document.getElementById('flashcards-filter-hsk')?.addEventListener('change', renderFlashcards);

const flashcardsScriptEl = document.getElementById('flashcards-script');
if (flashcardsScriptEl) {
  flashcardsScriptEl.value = cardScript;
  document.body.classList.toggle('script-trad', cardScript === 'trad');
  flashcardsScriptEl.addEventListener('change', () => {
    cardScript = flashcardsScriptEl.value === 'trad' ? 'trad' : 'simp';
    try { localStorage.setItem(SCRIPT_KEY, cardScript); } catch (e) {}
    document.body.classList.toggle('script-trad', cardScript === 'trad');
    renderFlashcards();
  });
}
document.getElementById('flashcards-show-all')?.addEventListener('click', () => {
  showAll = !showAll;
  flippedCards.clear();
  document.getElementById('toggle-label').textContent = showAll ? 'Study mode' : 'Show all';
  renderFlashcards();
});
document.getElementById('flashcards-clear')?.addEventListener('click', () => {
  const map = loadFlashcards();
  const count = Object.keys(map).length;
  if (count === 0) return;
  if (confirm(`Clear all ${count} flashcards? This can't be undone.`)) {
    saveFlashcards({}); flippedCards.clear(); renderFlashcards();
  }
});
document.getElementById('flashcards-export')?.addEventListener('click', () => {
  const list = flashcardsAsList(loadFlashcards());
  if (list.length === 0) return;
  // Exports whichever script is on display — someone studying Traditional wants
  // a Traditional Anki deck. Both forms are emitted so the file is unambiguous
  // about what it contains, and round-trips regardless of the toggle state.
  const lines = ['hanzi,traditional,pinyin,meaning'].concat(
    list.map(c => [cardDisplayHanzi(c), c.trad || c.hanzi, c.pinyin || '', c.meaning || '']
      .map(v => `"${String(v).replace(/"/g, '""')}"`).join(','))
  );
  const blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'flashcards.csv'; a.click();
  URL.revokeObjectURL(url);
});
window.addEventListener('storage', (e) => { if (e.key === FLASHCARDS_KEY) renderFlashcards(); });

// ============================================================
// ============================================================
// Accounts (Google OAuth + magic-link fallback)
// ============================================================
// Auth flow overview:
//   - On page load: GET /api/me → sets currentUser and updates nav pill
//   - Guest → "Sign in" button visible → click opens signin-modal
//   - Signed in → email + dropdown visible → dropdown has Sign out
//   - Google button → redirects to /api/auth/google → Google redirects back
//   - Email button → 2-step modal (enter email → enter code) hits
//     /api/auth/magic-link/{request,verify}, session cookie set on success
const authPill = document.getElementById('auth-pill');
const signinBtn = document.getElementById('signin-btn');
const userMenu = document.getElementById('user-menu');
const userMenuBtn = document.getElementById('user-menu-btn');
const userMenuEmail = document.getElementById('user-menu-email');
const userMenuDropdown = document.getElementById('user-menu-dropdown');
const signoutBtn = document.getElementById('signout-btn');
const signinModal = document.getElementById('signin-modal');
const signinStepPrimary = document.getElementById('signin-step-primary');
const signinStepEmail = document.getElementById('signin-step-email');
const signinStepCode = document.getElementById('signin-step-code');
const signinGoogleBtn = document.getElementById('signin-google-btn');
const signinEmailToggle = document.getElementById('signin-email-toggle');
const signinBackToPrimary = document.getElementById('signin-back-to-primary');
const signinBackToEmail = document.getElementById('signin-back-to-email');
const signinCancelBtn = document.getElementById('signin-cancel-btn');
const signinEmailInput = document.getElementById('signin-email');
const signinCodeInput = document.getElementById('signin-code');
const signinEmailEcho = document.getElementById('signin-email-echo');
const signinSendCodeBtn = document.getElementById('signin-send-code-btn');
const signinVerifyCodeBtn = document.getElementById('signin-verify-code-btn');
const signinError1 = document.getElementById('signin-error-1');
const signinError2 = document.getElementById('signin-error-2');

let currentUser = null;

function updateAuthPill() {
  if (!authPill) return;
  authPill.classList.remove('hidden');
  if (currentUser) {
    if (signinBtn) signinBtn.classList.add('hidden');
    if (userMenu) userMenu.classList.remove('hidden');
    if (userMenuEmail) userMenuEmail.textContent = currentUser.email;
  } else {
    if (signinBtn) signinBtn.classList.remove('hidden');
    if (userMenu) userMenu.classList.add('hidden');
  }
}

async function fetchCurrentUser() {
  try {
    const res = await fetch('/api/me', { credentials: 'same-origin' });
    if (!res.ok) return;
    const data = await res.json();
    currentUser = data.user || null;
    updateAuthPill();
    maybeNudgeClaim();
  } catch (e) {
    // Silent — being unable to check auth just leaves the pill hidden.
    console.warn('[auth] /api/me failed:', e);
  }
}

// Nudge the user to claim unclaimed browser chapters, ONCE per session, on the
// app page. Fires whether they signed in here OR arrived at /start already
// signed in (e.g. signed in on the front page, then walked over) — so the
// reminder follows them into the app instead of only firing on whichever page
// they happened to sign in on. The sessionStorage flag caps it to one open per
// session (a refresh / re-navigation won't keep re-popping it); claiming empties
// the browser bucket, so it stops on its own.
async function maybeNudgeClaim() {
  // Strip the one-time Google sign-in flag from the URL (cosmetic).
  try {
    const p = new URLSearchParams(location.search);
    if (p.get('signedin') === '1') {
      p.delete('signedin');
      const qs = p.toString();
      history.replaceState({}, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
    }
  } catch (e) {}
  if (!currentUser || jobHistory.length === 0) return;
  try { if (sessionStorage.getItem('startClaimNudged')) return; } catch (e) {}
  // Only nudge when there's genuinely something to claim (a bucket chapter not
  // already in the account) — otherwise we'd pop an empty drawer AND burn the
  // once-per-session flag so a later real claim gets no nudge.
  let accKeys = new Set();
  try {
    const r = await fetch('/api/library', { credentials: 'same-origin' });
    const body = await r.json().catch(() => ({}));
    if (body && body.signedIn && Array.isArray(body.library)) {
      accKeys = new Set(body.library.map(e => e.url + '|' + (e.level || 'hsk3')));
    } else { return; }
  } catch (e) { return; }
  if (!jobHistory.some(h => !accKeys.has(h.url + '|' + (h.level || 'hsk3')))) return;
  try { sessionStorage.setItem('startClaimNudged', '1'); } catch (e) {}
  openRecentDrawer();
}

function openSigninModal() {
  if (!signinModal) return;
  if (signinStepPrimary) signinStepPrimary.classList.remove('hidden');
  if (signinStepEmail) signinStepEmail.classList.add('hidden');
  if (signinStepCode) signinStepCode.classList.add('hidden');
  if (signinError1) signinError1.classList.add('hidden');
  if (signinError2) signinError2.classList.add('hidden');
  signinModal.classList.remove('hidden');
}
function closeSigninModal() { if (signinModal) signinModal.classList.add('hidden'); }

async function signinSendCode() {
  const email = signinEmailInput && signinEmailInput.value.trim();
  if (!email) { showSigninError1('Please enter your email.'); return; }
  signinSendCodeBtn.disabled = true; signinSendCodeBtn.textContent = 'Sending…';
  try {
    const res = await fetch('/api/auth/magic-link/request', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { showSigninError1(data.error || 'Could not send the code.'); return; }
    if (signinEmailEcho) signinEmailEcho.textContent = email;
    if (signinStepEmail) signinStepEmail.classList.add('hidden');
    if (signinStepCode) signinStepCode.classList.remove('hidden');
    if (signinCodeInput) setTimeout(() => signinCodeInput.focus(), 20);
  } catch (e) {
    showSigninError1('Network error. Please try again.');
  } finally {
    signinSendCodeBtn.disabled = false; signinSendCodeBtn.textContent = 'Send code';
  }
}

async function signinVerifyCode() {
  const email = signinEmailInput && signinEmailInput.value.trim();
  const code = signinCodeInput && signinCodeInput.value.trim();
  if (!code) { showSigninError2('Please enter the code.'); return; }
  signinVerifyCodeBtn.disabled = true; signinVerifyCodeBtn.textContent = 'Verifying…';
  try {
    const res = await fetch('/api/auth/magic-link/verify', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ email, code }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.user) { showSigninError2(data.error || 'Invalid code.'); return; }
    currentUser = data.user;
    updateAuthPill();
    closeSigninModal();
    maybeNudgeClaim(); // surface the claim step if there are unclaimed browser chapters
  } catch (e) {
    showSigninError2('Network error. Please try again.');
  } finally {
    signinVerifyCodeBtn.disabled = false; signinVerifyCodeBtn.textContent = 'Verify & sign in';
  }
}

async function signOut() {
  try {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' });
  } catch {}
  currentUser = null;
  updateAuthPill();
  if (userMenuDropdown) userMenuDropdown.classList.add('hidden');
  renderHistory(); // repaint the drawer to the anonymous bucket (drop the signed-out account's rows)
}

function showSigninError1(msg) {
  if (!signinError1) return;
  signinError1.textContent = msg;
  signinError1.classList.remove('hidden');
}
function showSigninError2(msg) {
  if (!signinError2) return;
  signinError2.textContent = msg;
  signinError2.classList.remove('hidden');
}

// Wire up the sign-in UI. All handlers guarded so missing elements don't throw.
if (signinBtn) signinBtn.addEventListener('click', openSigninModal);
if (signinCancelBtn) signinCancelBtn.addEventListener('click', closeSigninModal);
if (signinModal) signinModal.addEventListener('click', (e) => { if (e.target === signinModal) closeSigninModal(); });
if (signinGoogleBtn) signinGoogleBtn.addEventListener('click', () => { window.location.href = '/api/auth/google?return=' + encodeURIComponent(location.pathname); });
if (signinEmailToggle) signinEmailToggle.addEventListener('click', () => {
  if (signinStepPrimary) signinStepPrimary.classList.add('hidden');
  if (signinStepEmail) signinStepEmail.classList.remove('hidden');
  if (signinEmailInput) setTimeout(() => signinEmailInput.focus(), 20);
});
if (signinBackToPrimary) signinBackToPrimary.addEventListener('click', () => {
  if (signinStepEmail) signinStepEmail.classList.add('hidden');
  if (signinStepPrimary) signinStepPrimary.classList.remove('hidden');
});
if (signinBackToEmail) signinBackToEmail.addEventListener('click', () => {
  if (signinStepCode) signinStepCode.classList.add('hidden');
  if (signinStepEmail) signinStepEmail.classList.remove('hidden');
});
if (signinSendCodeBtn) signinSendCodeBtn.addEventListener('click', signinSendCode);
if (signinVerifyCodeBtn) signinVerifyCodeBtn.addEventListener('click', signinVerifyCode);
if (signinEmailInput) signinEmailInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); signinSendCode(); }
});
if (signinCodeInput) {
  signinCodeInput.addEventListener('input', () => { signinCodeInput.value = signinCodeInput.value.replace(/\D/g, ''); });
  signinCodeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); signinVerifyCode(); }
  });
}
if (userMenuBtn && userMenuDropdown) {
  userMenuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    userMenuDropdown.classList.toggle('hidden');
  });
  document.addEventListener('click', () => userMenuDropdown.classList.add('hidden'));
}
if (signoutBtn) signoutBtn.addEventListener('click', signOut);

// Kick off the initial /api/me check as soon as the script loads.
authReady = fetchCurrentUser();

// ============================================================
// Toast
// ============================================================
const toast = document.getElementById('toast');
function showToast(msg, ms = 4000) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), ms);
}

// ?level=hskN — preselect a level from a landing-page deep link.
// ?url=<chapter-url> — pre-fill the URL input (used when the user clicks a
//   synced-from-server chapter in Recent that doesn't have a reader URL yet).
(function handleLandingDeepLinks() {
  try {
    const params = new URLSearchParams(window.location.search);
    const levelParam = params.get('level');
    const urlParam = params.get('url');
    let dirty = false;

    if (levelParam && /^hsk[1-6]$/.test(levelParam)) {
      const radio = document.querySelector(`input[name="level"][value="${levelParam}"]`);
      if (radio) {
        radio.checked = true;
        radio.dispatchEvent(new Event('change', { bubbles: true }));
      }
      params.delete('level');
      dirty = true;
    }
    if (urlParam && urlRows) {
      // Replace the first (empty) URL row's value, or clear all rows and add
      // one row with the URL prefilled. Done via the existing setUrls helper
      // pattern: clear rows, append one with this value.
      try {
        urlRows.innerHTML = '';
        urlRows.appendChild(createUrlRow(urlParam));
      } catch (e) { console.error('[deep-link] url= prefill failed:', e); }
      params.delete('url');
      dirty = true;
    }
    if (dirty) {
      const newSearch = params.toString();
      const newUrl = window.location.pathname + (newSearch ? '?' + newSearch : '') + window.location.hash;
      history.replaceState({}, '', newUrl);
    }
  } catch (e) {
    console.error('[deep-link] failed to parse:', e);
  }
})();

// ============ Federated manga search ============
//
// Debounced query against /api/search. Renders a stack of cards, each with
// the series title (+ alt titles) and a row of source buttons. Clicking a
// source opens the series page on that site in a new tab so the user can
// pick a specific chapter and copy the URL back into the chapter-URL input.

// Per-source metadata. `recommended` = Chinese source, best simplification
// quality (no EN→CN translation step). `hasPaywalls` = paid chapters are
// COMMON enough on this site that users should expect to hit them.
// `noChapterList` = chapter-list endpoint not implemented for this source;
// clicking the button opens the series in a new tab instead of expanding.
// (Webtoons is excluded from paywall flag — they default to free with
// optional Fast Pass for early access, but the vast majority of chapters
// are accessible for free.)
const SEARCH_SOURCE_LABELS = {
  'manhuagui': { label: 'Manhuagui', recommended: true, hasPaywalls: false },
  'kuaikan': { label: 'KuaiKan', recommended: true, hasPaywalls: true, noChapterList: true },
  'webtoons-cn': { label: 'Webtoons CN', recommended: true, hasPaywalls: false },
  'webtoons-en': { label: 'Webtoons EN', recommended: false, hasPaywalls: false },
  'mangadex': { label: 'MangaDex', recommended: false, hasPaywalls: false },
  'tapas': { label: 'Tapas', recommended: false, hasPaywalls: true },
  'mangaplus': { label: 'MangaPlus', recommended: false, hasPaywalls: false, noChapterList: true },
  'dmzj': { label: 'DMZJ', recommended: true, hasPaywalls: false },
  'bilibili-manga': { label: 'Bilibili Manga', recommended: false, hasPaywalls: true },
};

const searchInput = document.getElementById('manga-search');
const searchResults = document.getElementById('search-results');
const searchHint = document.getElementById('search-hint');
let searchDebounce = null;
let lastSearchQuery = '';

function escapeHtmlSearch(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}

function renderSearchResults(results, query, translatorDone) {
  if (!results || results.length === 0) {
    searchResults.classList.remove('hidden');
    searchResults.innerHTML = '<div class="search-empty">No matches for "' + escapeHtmlSearch(query) + '". Try a different spelling or paste a chapter URL below.</div>';
    return;
  }
  const html = results.slice(0, 8).map(r => {
    // Cover images are routed through /api/proxy-image so we can send the
    // correct Referer header per CDN — most manga CDNs block direct hotlinks
    // from our origin. Server-side fetch + stream back bypasses this and
    // also adds a 1-hour image cache so repeat searches don't re-fetch.
    // Initials-placeholder is the final fallback if the proxy itself fails.
    const placeholderText = escapeHtmlSearch((r.title || '?').slice(0, 2).toUpperCase());
    const cover = r.coverUrl
      ? '<img class="search-card-cover" src="/api/proxy-image?url=' + encodeURIComponent(r.coverUrl) + '" alt="" loading="lazy" onerror="this.outerHTML=\'<div class=&quot;search-card-cover search-card-cover-fallback&quot;>' + placeholderText + '</div>\'">'
      : '<div class="search-card-cover search-card-cover-fallback">' + placeholderText + '</div>';
    // Only show CJK alt titles (Simplified/Traditional Chinese). The MangaDex
    // altTitles list also contains romanized Korean ("Jeonjijeok Dokja Sijeom"),
    // French ("Lecteur omniscient"), Polish, etc. — noise for a Chinese-reading
    // audience. Japanese (has kana) and Korean (Hangul) are also dropped: they
    // share Han characters but aren't usable as a Chinese-vocabulary aid.
    const isUsefulChineseAlt = t =>
      /[一-鿿]/.test(t) &&     // contains Han
      !/[぀-ゟ]/.test(t) &&    // not Hiragana (Japanese)
      !/[゠-ヿ]/.test(t) &&    // not Katakana (Japanese)
      !/[가-힯]/.test(t);      // not Hangul (Korean)
    const alts = (r.altTitles || [])
      .filter(t => t !== r.title && isUsefulChineseAlt(t))
      .slice(0, 3)
      .join(' / ');
    // CN-quality preference: when a bucket has ONLY EN sources and the
    // translator step hasn't finished yet, hold the source-button area in
    // a "Checking for Chinese version…" state. This avoids the user clicking
    // EN in the ~1s window before the CN sibling merges into the bucket —
    // CN→simplified-CN is unambiguously a better translation chain than
    // EN→CN→simplified-CN. If translatorDone resolves with no CN found, the
    // EN button shows up normally.
    const hasCn = r.sources.some(s => s.source === 'webtoons-cn');
    const hasEn = r.sources.some(s => s.source === 'webtoons-en');
    const showCnPlaceholder = hasEn && !hasCn && !translatorDone;
    // When CN exists alongside EN, only show CN. EN→CN→simple-CN is a strictly
    // worse chain than CN→simple-CN. We surface CN as the only path; the EN
    // version is still reachable on Webtoons.com directly if anyone wants it.
    const visibleSources = hasCn ? r.sources.filter(s => s.source !== 'webtoons-en') : r.sources;
    const sourceButtons = showCnPlaceholder
      ? '<div class="search-source-pending">Checking for Chinese version…</div>'
      : visibleSources.map((s, sIdx) => {
          const meta = SEARCH_SOURCE_LABELS[s.source] || { label: s.source, recommended: false, hasPaywalls: false };
          const cls = 'search-source-btn' + (meta.recommended ? ' recommended' : '');
          const cnBadge = meta.recommended ? ' <span class="badge">CN</span>' : '';
          const payBadge = meta.hasPaywalls
            ? ' <span class="badge badge-pay" title="Some chapters on this site may require payment">$</span>'
            : '';
          // Source buttons are PRIMARY actions. For sources where we have chapter
          // list, click expands the list inline. For sources without (KuaiKan,
          // MangaPlus — JS-rendered SPAs we can't scrape), click opens the
          // series page in a new tab as a fallback.
          if (meta.noChapterList) {
            return '<a class="' + cls + '" href="' + escapeHtmlSearch(s.seriesUrl) + '" target="_blank" rel="noopener" title="Opens in new tab — chapter browsing not yet supported for this source">' + escapeHtmlSearch(meta.label) + cnBadge + payBadge + ' <span class="ext-icon">↗</span></a>';
          }
          return '<button type="button" class="' + cls + ' js-show-chapters" data-source="' + escapeHtmlSearch(s.source) + '" data-series-url="' + escapeHtmlSearch(s.seriesUrl) + '">' + escapeHtmlSearch(meta.label) + cnBadge + payBadge + '</button>';
        }).join('');
    // If ALL sources for this series have paywalls, surface a hint so users
    // know to look for free chapters before assuming the series is accessible.
    const allPaid = r.sources.every(s => (SEARCH_SOURCE_LABELS[s.source] || {}).hasPaywalls);
    const paywallNote = allPaid && r.sources.length > 0
      ? '<div class="search-card-note">Some or all chapters on these sources may require payment.</div>'
      : '';
    return '<div class="search-card">' + cover + '<div class="search-card-body">' +
      '<div class="search-card-title">' + escapeHtmlSearch(r.title) + '</div>' +
      (alts ? '<div class="search-card-alts">' + escapeHtmlSearch(alts) + '</div>' : '') +
      '<div class="search-card-sources">' + sourceButtons + '</div>' +
      paywallNote +
      '<div class="search-card-chapters hidden"></div>' +
    '</div></div>';
  }).join('');
  searchResults.classList.remove('hidden');
  searchResults.innerHTML = html;
}

let activeSearchSource = null;

function performSearch(query) {
  if (query === lastSearchQuery) return;
  lastSearchQuery = query;
  // Cancel any in-flight stream before starting a new one — keeps the
  // race-condition window small and avoids paying for stale results.
  if (activeSearchSource) { try { activeSearchSource.close(); } catch {} activeSearchSource = null; }

  if (query.length < 2) {
    searchResults.classList.add('hidden');
    searchResults.innerHTML = '';
    return;
  }
  searchResults.classList.remove('hidden');
  searchResults.innerHTML = '<div class="search-loading">Searching…</div>';

  // SSE: fast sources paint immediately while slow sources (Manhuagui, DMZJ)
  // either show up later or never block the response. Each snapshot event
  // replaces the rendered list with the latest aggregated results.
  const source = new EventSource('/api/search/stream?q=' + encodeURIComponent(query));
  activeSearchSource = source;
  let hadAny = false;

  source.addEventListener('snapshot', ev => {
    if (query !== lastSearchQuery) { source.close(); return; }
    try {
      const data = JSON.parse(ev.data);
      hadAny = true;
      renderSearchResults(data.results || [], query, !!data.translatorDone);
    } catch (e) { console.warn('[search-stream] snapshot parse failed', e); }
  });
  source.addEventListener('done', () => {
    source.close();
    if (activeSearchSource === source) activeSearchSource = null;
    if (!hadAny && query === lastSearchQuery) {
      searchResults.innerHTML = '<div class="search-empty">No matches for "' + escapeHtmlSearch(query) + '".</div>';
    }
  });
  source.addEventListener('error', () => {
    // EventSource fires `error` when WE call .close() too. Detect that case
    // (readyState === CLOSED) and silently skip — we don't want to wipe the
    // user's already-rendered results (or an open chapter textbox) just
    // because they clicked a source button mid-stream.
    const wasManuallyClosed = source.readyState === 2;
    source.close();
    if (activeSearchSource === source) activeSearchSource = null;
    if (wasManuallyClosed) return;
    if (!hadAny && query === lastSearchQuery) {
      searchResults.innerHTML = '<div class="search-empty">Search failed. Restart the server if the search endpoint is new.</div>';
    }
  });
}

if (searchInput) {
  searchInput.addEventListener('input', () => {
    const q = searchInput.value.trim();
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => performSearch(q), 350);
  });
}

// ============ File upload drop zone ============
//
// Wires the drop zone to /api/upload-images. On success, the response shape
// matches /api/process so the existing job-polling code in this file picks
// up the job and renders the "Processing..." run-view exactly like a
// URL-submitted job.

const uploadZone = document.getElementById('upload-zone');
const uploadInput = document.getElementById('upload-input');
const uploadFolderInput = document.getElementById('upload-folder-input');
const uploadFolderLink = document.getElementById('upload-folder-link');
const uploadStatus = document.getElementById('upload-status');

// Recursively walk a DirectoryEntry (from webkitGetAsEntry) into a flat File[].
// Chrome's readEntries returns at most ~100 entries per call, so we loop until
// it returns an empty batch.
async function walkEntry(entry, accumulator) {
  if (entry.isFile) {
    await new Promise(resolve => entry.file(f => { accumulator.push(f); resolve(); }, () => resolve()));
    return;
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const allEntries = [];
    while (true) {
      const batch = await new Promise(resolve => reader.readEntries(resolve, () => resolve([])));
      if (!batch.length) break;
      allEntries.push(...batch);
    }
    await Promise.all(allEntries.map(e => walkEntry(e, accumulator)));
  }
}

// Pull a flat File[] out of a DataTransfer, expanding any dropped folders.
// Falls back to dataTransfer.files when the items/entry API isn't available.
async function collectFilesFromDataTransfer(dt) {
  if (dt.items && dt.items.length && typeof dt.items[0].webkitGetAsEntry === 'function') {
    const files = [];
    const entries = [];
    for (let i = 0; i < dt.items.length; i++) {
      const e = dt.items[i].webkitGetAsEntry && dt.items[i].webkitGetAsEntry();
      if (e) entries.push(e);
    }
    await Promise.all(entries.map(e => walkEntry(e, files)));
    return files;
  }
  return Array.from(dt.files || []);
}

function setUploadStatus(msg, cls) {
  if (!uploadStatus) return;
  uploadStatus.textContent = msg;
  uploadStatus.className = 'upload-status ' + (cls || '');
}

async function handleUploadedFiles(fileList) {
  const all = Array.from(fileList);
  // Treat the OS-junk siblings folders often carry as "expected skips" — they
  // don't warrant a warning. Anything else the user-visible filter drops is
  // counted in `skippedReal` and surfaced so unexpected omissions (PSDs,
  // RAWs, sketch files) don't disappear silently.
  const isExpectedJunk = name => /(^|\/)(\.DS_Store|Thumbs\.db|desktop\.ini|\._.*)$/i.test(name);
  const files = all.filter(f => /^image\/(jpe?g|png|webp|gif)$/i.test(f.type));
  const skippedReal = all.filter(f =>
    !/^image\/(jpe?g|png|webp|gif)$/i.test(f.type) &&
    !isExpectedJunk(f.webkitRelativePath || f.name)
  ).length;
  // Sort so chapter pages upload in reading order. Prefer the full relative
  // path when present (folder uploads expose it on webkitRelativePath) so
  // nested folders sort sensibly; fall back to plain name. Numeric-aware
  // compare so "page2" < "page10".
  files.sort((a, b) => {
    const aKey = a.webkitRelativePath || a.name;
    const bKey = b.webkitRelativePath || b.name;
    return aKey.localeCompare(bKey, undefined, { numeric: true, sensitivity: 'base' });
  });
  if (files.length === 0) {
    setUploadStatus('No supported image files found. Drop jpg/png/webp/gif.', 'error');
    return;
  }
  if (files.length > 300) {
    setUploadStatus(`${files.length} files — over the 300-image limit. Split into multiple uploads.`, 'error');
    return;
  }
  // Read current form values for level + language
  const level = document.querySelector('input[name="level"]:checked')?.value || 'hsk3';
  const lang = inputLanguage || 'auto';
  const fd = new FormData();
  for (const f of files) fd.append('images', f, f.name);
  fd.append('level', level);
  fd.append('inputLanguage', lang);
  const skipNote = skippedReal > 0 ? ` (skipped ${skippedReal} non-image file${skippedReal === 1 ? '' : 's'})` : '';
  setUploadStatus(`Uploading ${files.length} images…${skipNote}`, 'uploading');
  try {
    const res = await fetch('/api/upload-images', { method: 'POST', body: fd, credentials: 'include' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setUploadStatus(err.error || `Upload failed (HTTP ${res.status}).`, 'error');
      return;
    }
    const data = await res.json();
    const jobIds = (data.jobs || []).map(j => j.jobId);
    if (jobIds.length === 0) {
      setUploadStatus('Upload succeeded but no jobs returned.', 'error');
      return;
    }
    // Hand off to the existing job-pollling flow.
    setUploadStatus('', '');
    uploadStatus.classList.add('hidden');
    // Replicate what handleFormSubmit does after createJob.
    queue = jobIds.map((id, i) => ({
      jobId: id,
      url: (data.jobs[i] && data.jobs[i].url) || `upload (${files.length} images)`,
      status: 'pending',
      message: 'Queued...',
    }));
    showView('run');
    renderRunJobs();
    if (!pollInterval) startPolling();
  } catch (e) {
    setUploadStatus('Upload failed: ' + (e && e.message ? e.message : 'unknown error'), 'error');
  }
}

if (uploadZone && uploadInput) {
  uploadZone.addEventListener('click', () => uploadInput.click());
  uploadZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); uploadInput.click(); } });
  uploadInput.addEventListener('change', () => { if (uploadInput.files?.length) handleUploadedFiles(uploadInput.files); });
  // Drag-drop. Have to stop default on dragover or drop won't fire.
  ['dragenter', 'dragover'].forEach(ev => uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(ev => uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.remove('dragging'); }));
  uploadZone.addEventListener('drop', async e => {
    e.preventDefault();
    if (!e.dataTransfer) return;
    // Snapshot items synchronously — DataTransferItemList becomes empty after
    // the event handler returns, so we have to walk it before any await.
    setUploadStatus('Reading dropped files…', 'uploading');
    try {
      const files = await collectFilesFromDataTransfer(e.dataTransfer);
      if (files.length) handleUploadedFiles(files);
      else setUploadStatus('No files found in drop. Try selecting them directly.', 'error');
    } catch (err) {
      setUploadStatus('Could not read dropped folder: ' + (err && err.message || 'unknown error'), 'error');
    }
  });
}

if (uploadFolderLink && uploadFolderInput) {
  uploadFolderLink.addEventListener('click', () => uploadFolderInput.click());
  uploadFolderInput.addEventListener('change', () => { if (uploadFolderInput.files?.length) handleUploadedFiles(uploadFolderInput.files); });
}

// ============ Chapter list expansion ============
//
// When the user clicks a source button on a search result card, fetch that
// source's chapter list and render it inline beneath the card. Clicking a
// chapter title processes it (same flow as the URL input). Paid/locked
// chapters get a $ icon and a clear "we can't process this" warning.

const chapterCache = new Map(); // source+url → chapters[]

function renderChapterList(container, source, seriesUrl, chapters) {
  // Drop paid chapters entirely — they're misleading. Payment on the source
  // site doesn't transfer to our server, so we can't process them.
  const lockedCount = chapters.filter(ch => ch.locked).length;
  const free = chapters.filter(ch => !ch.locked);
  if (free.length === 0) {
    if (lockedCount > 0) {
      container.innerHTML = '<div class="chapter-list-empty">All ' + lockedCount + ' chapters on this source require payment. We can\'t process paid chapters — try a different source, or use the upload field below for chapters you can view in your browser.</div>';
    } else {
      container.innerHTML = '<div class="chapter-list-empty">No chapters listed.</div>';
    }
    return;
  }
  // Compute the available chapter-number range. We trust the source's
  // `number` field (handles .5 chapters that some sites count as their own
  // sequential chapter). parseFloat for sort + range, but display the raw
  // string so "1.5" shows as "1.5" not "1.5000".
  const numbers = free.map(c => parseFloat(c.number || '0')).filter(n => isFinite(n));
  const minN = numbers.length ? Math.min(...numbers) : 1;
  const maxN = numbers.length ? Math.max(...numbers) : free.length;
  // Format: integer if no decimals, otherwise as-is. "1.5" stays "1.5"; "1" stays "1".
  const fmt = (n) => Number.isInteger(n) ? String(n) : String(n);
  const range = numbers.length ? fmt(minN) + '–' + fmt(maxN) : '?';
  const note = lockedCount > 0
    ? '<div class="chapter-list-hint">' + lockedCount + ' paid chapters hidden.</div>'
    : '';
  container.innerHTML = `
    <div class="chapter-input-row">
      <label class="chapter-input-label">Chapter <span class="chapter-input-range">(${range})</span></label>
      <div class="chapter-input-controls">
        <input type="text" class="chapter-input text-input" data-source="${escapeHtmlSearch(source)}" data-series-url="${escapeHtmlSearch(seriesUrl)}" placeholder="${fmt(maxN)}" inputmode="decimal" autocomplete="off">
        <button type="button" class="chapter-input-go btn-primary">Read</button>
      </div>
      <div class="chapter-input-status"></div>
    </div>
    ${note}
  `;
}

// Render the chapter textbox IMMEDIATELY (no network wait). For Webtoons we
// can construct chapter URLs from input alone (episode_no in URL); for other
// sources we kick off a background chapter-list fetch to populate the range
// hint. The textbox is rendered ONCE — subsequent data updates only change
// the range-label text, so the user can start typing immediately and never
// loses focus / typed input mid-fetch.
function showChaptersFor(card, source, seriesUrl) {
  const container = card.querySelector('.search-card-chapters');
  if (!container) return;
  const key = source + '::' + seriesUrl;
  // Toggle: if same source already open, close it.
  if (container.dataset.activeKey === key && !container.classList.contains('hidden')) {
    container.classList.add('hidden');
    container.dataset.activeKey = '';
    return;
  }
  container.dataset.activeKey = key;
  container.classList.remove('hidden');

  // Render once with "loading…" while the metadata fetch resolves; we
  // only update the range label afterward so the input doesn't lose focus.
  renderChapterTextbox(container, source, seriesUrl, '(loading…)');

  const isWebtoons = source === 'webtoons-en' || source === 'webtoons-cn';

  // Webtoons: lightweight path. One HTTP request returns the max episode_no
  // (the latest chapter and the upper bound of our typeable range). Chapter
  // URLs are built directly from episode_no on the frontend below, so we
  // never need to fetch the full chapter list. ~800ms vs ~10s for long
  // series like Tower of God.
  if (isWebtoons) {
    fetch('/api/webtoons-chapter-count?url=' + encodeURIComponent(seriesUrl))
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (container.dataset.activeKey !== key) return;
        const rangeSpan = container.querySelector('.chapter-input-range');
        const input = container.querySelector('.chapter-input');
        const count = data && data.count;
        if (count && rangeSpan) {
          rangeSpan.textContent = '(1–' + count + ')';
          if (input) {
            input.placeholder = String(count);
            input.dataset.maxChapter = String(count);
          }
        } else if (rangeSpan) {
          rangeSpan.textContent = '(any chapter)';
        }
      })
      .catch(() => {
        const rangeSpan = container.querySelector('.chapter-input-range');
        if (rangeSpan) rangeSpan.textContent = '(any chapter)';
      });
    return;
  }

  // Non-Webtoons: full chapter-list fetch (only used for sources that
  // expose getChapterList — currently none are wired into onsite search,
  // but the code path stays in case we surface them later).
  const updateRange = (chapters) => {
    if (container.dataset.activeKey !== key) return;
    const free = chapters.filter(c => !c.locked);
    const numbers = free
      .map(c => parseFloat(c.number || ''))
      .filter(n => isFinite(n));
    const rangeSpan = container.querySelector('.chapter-input-range');
    const input = container.querySelector('.chapter-input');
    if (numbers.length) {
      const min = Math.min(...numbers);
      const max = Math.max(...numbers);
      if (rangeSpan) rangeSpan.textContent = '(' + min + '–' + max + ')';
      if (input) input.placeholder = String(max);
    } else {
      if (rangeSpan) rangeSpan.textContent = '(?)';
    }
  };
  if (chapterCache.has(key)) {
    updateRange(chapterCache.get(key));
    return;
  }
  fetch('/api/chapter-list?source=' + encodeURIComponent(source) + '&url=' + encodeURIComponent(seriesUrl))
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data) return;
      const chapters = data.chapters || [];
      chapterCache.set(key, chapters);
      updateRange(chapters);
    })
    .catch(() => {
      const rangeSpan = container.querySelector('.chapter-input-range');
      if (rangeSpan) rangeSpan.textContent = '(?)';
    });
}

function renderChapterTextbox(container, source, seriesUrl, rangeStr) {
  // No "Read" button — the form already has a Simplify-chapter submit button
  // below. The textbox WRITES INTO the existing URL input as the user types,
  // so the user can verify the URL before clicking the existing submit button.
  // Status line surfaces the constructed URL (or any error) for verification.
  // Each Add click APPENDS to the URL box — repeated clicks build up a
  // list of chapters (across one source or many). Duplicate URLs are
  // skipped so re-clicking Add with the same value is a no-op.
  container.innerHTML = `
    <div class="chapter-input-row">
      <label class="chapter-input-label">Chapter <span class="chapter-input-range">${rangeStr}</span></label>
      <div class="chapter-input-controls">
        <input type="text" class="chapter-input" data-source="${escapeHtmlSearch(source)}" data-series-url="${escapeHtmlSearch(seriesUrl)}" placeholder="1 or 1-10" inputmode="text" autocomplete="off" tabindex="0">
        <button type="button" class="chapter-input-add btn-primary" disabled>Add chapter</button>
      </div>
      <div class="chapter-input-status"></div>
      <a class="chapter-input-verify hidden" target="_blank" rel="noopener noreferrer">Open chapter to verify ↗</a>
    </div>
  `;
  const input = container.querySelector('.chapter-input');
  const addBtn = container.querySelector('.chapter-input-add');
  const row = container.querySelector('.chapter-input-row');
  const status = container.querySelector('.chapter-input-status');

  // Defer focus to next tick so it runs after any synchronous layout/SSE
  // side-effects. Click anywhere in the row re-focuses the input.
  if (input) setTimeout(() => input.focus(), 0);
  if (row && input) row.addEventListener('click', e => { if (!e.target.closest('button')) input.focus(); });

  // Validate as the user types so the Add button reflects what would happen.
  // Parsing is local (no network); we only resolve URLs when Add is clicked.
  if (input && addBtn) {
    input.addEventListener('input', () => {
      const v = input.value.trim();
      const parsed = parseChapterRange(v);
      // Re-typing dismisses any open multi-match picker from the previous Add.
      container.querySelectorAll('.chapter-picker').forEach(el => el.remove());
      if (!v) {
        addBtn.disabled = true;
        addBtn.textContent = 'Add chapter';
        status.textContent = '';
        status.className = 'chapter-input-status';
        const verifyLink = container.querySelector('.chapter-input-verify');
        if (verifyLink) { verifyLink.classList.add('hidden'); verifyLink.removeAttribute('href'); }
        return;
      }
      if (parsed.type === 'invalid') {
        addBtn.disabled = true;
        addBtn.textContent = 'Add chapter';
        status.textContent = 'Enter a chapter number (e.g. "5") or a range (e.g. "5-10").';
        status.className = 'chapter-input-status error';
        return;
      }
      const n = parsed.list.length;
      if (n > MAX_URLS) {
        addBtn.disabled = true;
        addBtn.textContent = 'Add ' + n + ' chapters';
        status.textContent = 'Range too large. Max ' + MAX_URLS + ' chapters per submission.';
        status.className = 'chapter-input-status error';
        return;
      }
      addBtn.disabled = false;
      addBtn.textContent = n === 1 ? 'Add chapter' : 'Add ' + n + ' chapters';
      status.textContent = '';
      status.className = 'chapter-input-status';
    });

    addBtn.addEventListener('click', () => {
      const v = input.value.trim();
      if (!v) return;
      resolveAndFillChapterUrl(source, seriesUrl, v, status, addBtn);
    });

    // Enter inside the input triggers Add when it's a valid range.
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !addBtn.disabled) {
        e.preventDefault();
        addBtn.click();
      }
    });
  }
}

// Resolve a single chapter or a range ("5" / "5-10") to URLs and append
// them to the URL box. Called when the user clicks the Add button next to
// the chapter input — not on every keystroke. Each click appends; the user
// reviews the URLs and clicks the existing Simplify-chapter button to
// process them.
async function resolveAndFillChapterUrl(source, seriesUrl, value, status, addBtn) {
  const row = status && status.parentElement;
  const verifyLink = row && row.querySelector('.chapter-input-verify');
  const setStatus = (msg, cls) => { if (status) { status.textContent = msg; status.className = 'chapter-input-status ' + (cls || ''); } };
  const hideVerify = () => { if (verifyLink) { verifyLink.classList.add('hidden'); verifyLink.removeAttribute('href'); } };
  const showVerify = (url) => { if (verifyLink) { verifyLink.href = url; verifyLink.classList.remove('hidden'); } };

  hideVerify();

  const parsed = parseChapterRange(value);
  if (parsed.type === 'invalid') {
    setStatus('Enter a chapter number (e.g. "5") or a range (e.g. "5-10").', 'error');
    return;
  }
  const numbers = parsed.list;
  if (numbers.length > MAX_URLS) {
    setStatus('Range too large. Max ' + MAX_URLS + ' chapters per submission — try a smaller range like "1-' + MAX_URLS + '".', 'error');
    return;
  }

  if (addBtn) addBtn.disabled = true;

  // Webtoons: chapter URLs are built directly from episode_no — no server
  // call needed. Validate against the cached max chapter count if we have
  // one so we catch out-of-range numbers up front. ~0ms vs the multi-page
  // /api/chapter-by-number lookup the non-Webtoons branch does below.
  const isWebtoons = source === 'webtoons-en' || source === 'webtoons-cn';
  if (isWebtoons) {
    const inputEl = row && row.querySelector('.chapter-input');
    const maxChapter = inputEl && inputEl.dataset.maxChapter
      ? parseInt(inputEl.dataset.maxChapter, 10)
      : null;
    const urls = [];
    const errors = [];
    for (const n of numbers) {
      const epNo = parseInt(n, 10);
      if (!isFinite(epNo) || epNo < 1) { errors.push('Chapter ' + n + ' is not a valid episode number.'); continue; }
      if (maxChapter && epNo > maxChapter) { errors.push('Chapter ' + n + ' is past the latest (' + maxChapter + ').'); continue; }
      const chapterUrl = buildWebtoonsChapterUrl(seriesUrl, epNo);
      if (!chapterUrl) { errors.push('Could not build URL for chapter ' + n + '.'); continue; }
      urls.push(chapterUrl);
    }
    if (urls.length === 0) {
      if (addBtn) addBtn.disabled = false;
      setStatus(errors[0] || 'Could not add chapters.', 'error');
      return;
    }
    const { added, dropped, duplicates } = appendChapterUrlsToBox(urls);
    let msg;
    if (added === 0) {
      msg = duplicates > 0 ? 'Already added — these chapters are in the list below.' : 'Could not add chapters.';
    } else {
      msg = added === 1
        ? 'Added below. Press Simplify chapter to process.'
        : 'Added ' + added + ' chapters below. Press Simplify chapter to process them all.';
    }
    if (duplicates > 0 && added > 0) msg += ' (' + duplicates + ' already in list, skipped.)';
    if (errors.length) msg += ' ' + errors.length + ' invalid.';
    if (dropped > 0) msg += ' ' + dropped + ' couldn\'t fit — clear another URL row.';
    setStatus(msg, (dropped > 0 || added === 0) ? 'error' : 'ok');
    if (added === 1 && urls.length === 1) showVerify(urls[0]);
    if (addBtn) addBtn.disabled = false;
    return;
  }

  const loadingMsg = numbers.length === 1
    ? 'Resolving chapter ' + numbers[0] + '…'
    : 'Resolving chapters ' + numbers[0] + '–' + numbers[numbers.length - 1] + '…';
  setStatus(loadingMsg, 'loading');

  // Non-Webtoons: resolve each chapter in parallel against the chapter list
  // via /api/chapter-by-number. Each response carries data.matches[] when
  // the typed number matched multiple chapters (multi-season collision) —
  // for single-chapter adds we surface those as a picker; for ranges we
  // auto-pick newest with a warning.
  const results = await Promise.all(numbers.map(async n => {
    try {
      const res = await fetch('/api/chapter-by-number?source=' + encodeURIComponent(source) + '&url=' + encodeURIComponent(seriesUrl) + '&n=' + encodeURIComponent(n));
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        return { n, error: err.error || 'Chapter ' + n + ' not found' };
      }
      const data = await res.json();
      if (data.locked) return { n, locked: true };
      return { n, chapterUrl: data.chapterUrl, matches: data.matches };
    } catch (e) {
      return { n, error: 'Lookup failed' };
    }
  }));

  const ok = results.filter(r => r.chapterUrl);
  const locked = results.filter(r => r.locked);
  const errored = results.filter(r => r.error);

  if (ok.length === 0) {
    if (addBtn) addBtn.disabled = false;
    setStatus(errored[0]?.error || (locked.length ? 'Those chapters are paid only.' : 'Could not find any chapters.'), 'error');
    return;
  }

  // Single-chapter add with multiple matches → show picker, don't auto-add.
  // This is the multi-season disambiguation path: user typed "5" and the
  // chapter list has S1 Ep. 5, S2 Ep. 5, S3 Ep. 5; let them pick.
  if (numbers.length === 1 && ok[0].matches && ok[0].matches.length > 1) {
    showChapterPicker(ok[0].matches, status, addBtn, ok[0].n);
    return;
  }

  const { added, dropped, duplicates } = appendChapterUrlsToBox(ok.map(r => r.chapterUrl));

  // Status: count + skip notices. Verify link only makes sense when a single
  // new URL was just added (so opening it verifies what was just added).
  let msg;
  if (added === 0) {
    msg = duplicates > 0 ? 'Already added — these chapters are in the list below.' : 'Could not add chapters.';
  } else {
    msg = added === 1
      ? 'Added below. Press Simplify chapter to process.'
      : 'Added ' + added + ' chapters below. Press Simplify chapter to process them all.';
  }
  if (duplicates > 0 && added > 0) msg += ' (' + duplicates + ' already in list, skipped.)';
  if (locked.length) msg += ' Skipped ' + locked.length + ' paid chapter' + (locked.length === 1 ? '' : 's') + '.';
  if (errored.length) msg += ' ' + errored.length + ' not found.';
  if (dropped > 0) msg += ' ' + dropped + ' couldn\'t fit — clear another URL row.';
  // For ranges, warn when any chapter had multi-season collisions — we used the
  // newest match silently (the most likely intent for batch operations).
  const ambiguous = ok.filter(r => r.matches && r.matches.length > 1).length;
  if (ambiguous > 0) msg += ' ' + ambiguous + ' chapter' + (ambiguous === 1 ? '' : 's') + ' had multiple matches; used newest season.';
  setStatus(msg, (dropped > 0 || added === 0) ? 'error' : 'ok');
  if (added === 1 && ok.length === 1) showVerify(ok[0].chapterUrl);
  // Re-enable Add — re-clicking with the same value is a duplicate-skip no-op,
  // editing the value enables/disables via the input event handler.
  if (addBtn) addBtn.disabled = false;
}

// Render an inline picker under the chapter input when a typed chapter number
// matches multiple chapters (multi-season collision). User clicks one option
// to add THAT specific chapter URL.
function showChapterPicker(matches, status, addBtn, typedNumber) {
  const row = status && status.parentElement;
  if (!row) return;
  // Remove any existing picker from a previous click.
  row.querySelectorAll('.chapter-picker').forEach(el => el.remove());

  const picker = document.createElement('div');
  picker.className = 'chapter-picker';
  picker.innerHTML =
    '<div class="chapter-picker-label">"' + typedNumber + '" matches ' + matches.length + ' chapters — pick one:</div>' +
    '<div class="chapter-picker-options">' +
    matches.map((m, i) =>
      '<button type="button" class="chapter-picker-option" data-idx="' + i + '">' +
      escapeHtmlSearch(m.title) +
      '</button>'
    ).join('') +
    '</div>';
  row.appendChild(picker);

  // Re-enable Add immediately so user can re-type to dismiss.
  if (addBtn) addBtn.disabled = false;
  status.textContent = '';
  status.className = 'chapter-input-status';

  picker.addEventListener('click', e => {
    const btn = e.target.closest('.chapter-picker-option');
    if (!btn) return;
    const idx = parseInt(btn.dataset.idx, 10);
    const pick = matches[idx];
    if (!pick) return;
    picker.remove();
    const { added, dropped, duplicates } = appendChapterUrlsToBox([pick.chapterUrl]);
    let msg;
    if (added === 0 && duplicates > 0) {
      msg = 'Already added — that chapter is in the list below.';
    } else if (added === 0) {
      msg = 'Could not add chapter.';
    } else {
      msg = 'Added "' + pick.title + '" below. Press Simplify chapter to process.';
    }
    if (dropped > 0) msg += ' Could not fit — clear another URL row.';
    status.textContent = msg;
    status.className = 'chapter-input-status ' + ((dropped > 0 || added === 0) ? 'error' : 'ok');
    // Show verify for the added chapter
    if (added === 1) {
      const verifyLink = row.querySelector('.chapter-input-verify');
      if (verifyLink) { verifyLink.href = pick.chapterUrl; verifyLink.classList.remove('hidden'); }
    }
  });
}

if (searchResults) {
  searchResults.addEventListener('click', e => {
    const sourceBtn = e.target.closest('.js-show-chapters');
    if (sourceBtn) {
      // Close any in-flight search stream — otherwise its next snapshot will
      // replace the entire search-results DOM (and the textbox we're about
      // to open) when it arrives a moment later.
      if (activeSearchSource) { try { activeSearchSource.close(); } catch {} activeSearchSource = null; }
      const card = sourceBtn.closest('.search-card');
      const source = sourceBtn.dataset.source;
      const seriesUrl = sourceBtn.dataset.seriesUrl;
      if (card && source && seriesUrl) showChaptersFor(card, source, seriesUrl);
      return;
    }
  });

  // (Enter-in-chapter-input handler lives in renderChapterTextbox now — it
  // triggers the Add button instead of the form submit, since adding URLs
  // to the box is the explicit user action.)
}

// Construct a Webtoons viewer URL directly from a series URL + episode_no.
// Webtoons accepts ANY slug in the viewer path as long as title_no and
// episode_no in the query string are valid — we use /x/viewer so we don't
// have to know the real per-chapter slug.
//
// We use this instead of looking up chapter titles because:
//   - episode_no is globally unique within a series (no multi-season collision)
//   - no chapter-list scrape needed (the latest episode_no comes from a single
//     page-1 fetch via /api/webtoons-chapter-count)
//   - chapters with non-numeric titles ("Prologue", "Special") are still
//     addressable by their episode_no
// Trade-off: episode_no doesn't match the season-relative number a reader
// sees on the Webtoons chapter list (e.g. "(S2) Ep. 5" is episode_no=80-ish,
// not 5). The verify link mitigates this — user clicks to confirm what URL
// they'd be processing before submitting.
function buildWebtoonsChapterUrl(seriesUrl, episodeNo) {
  try {
    const u = new URL(seriesUrl);
    const titleNoMatch = u.search.match(/title_no=(\d+)/);
    if (!titleNoMatch) return null;
    const titleNo = titleNoMatch[1];
    const path = u.pathname.replace(/\/list\/?$/, '');
    const n = parseInt(episodeNo, 10);
    if (!isFinite(n) || n < 1) return null;
    return u.origin + path + '/x/viewer?title_no=' + titleNo + '&episode_no=' + n;
  } catch { return null; }
}

// ============ Recommended picks (start page) ============
// Hand-curated, edited here. Covers are fetched through /api/proxy-image because
// the Webtoons CDN rejects hotlinks without a webtoons.com Referer. Clicking a
// tile fills the chapter-URL row with episode 1 — the same path the ?url= deep
// link uses — so the rest of the form (level, Read) works unchanged.
const RECOMMENDED_SERIES = [
  { id: 'lookism', zh: '看臉時代', en: 'Lookism', genre: 'action', titleNo: 546, path: '/zh-hant/school/kanlianshidai',
    cover: 'https://swebtoon-phinf.pstatic.net/20250204_11/1738644240839oXbWm_JPEG/546.jpg?type=crop540_540' },
  { id: 'orv', zh: '全知讀者視角', en: 'Omniscient Reader', genre: 'fantasy', titleNo: 2089, path: '/zh-hant/fantasy/duzhe',
    cover: 'https://swebtoon-phinf.pstatic.net/20250204_132/17386452652222A82u_JPEG/2089.jpg?type=crop540_540' },
  { id: 'tged', zh: '史詩級領地設計師', en: 'The Greatest Estate Developer', genre: 'comedy', titleNo: 4354, path: '/zh-hant/fantasy/estatedeveloper',
    cover: 'https://swebtoon-phinf.pstatic.net/20250205_118/1738719296488KUai4_PNG/4354.png?type=crop540_540' },
  { id: 'bastard', zh: '不肖子', en: 'Bastard', genre: 'thriller', titleNo: 631, path: '/zh-hant/thriller/buxiaozi',
    cover: 'https://swebtoon-phinf.pstatic.net/20250204_1/1738644247102vqtGS_JPEG/631.jpg?type=crop540_540' },
  { id: 'otl', zh: '純情大作戰', en: 'Operation: True Love', genre: 'romance', titleNo: 4823, path: '/zh-hant/romance/operation-true-love',
    cover: 'https://swebtoon-phinf.pstatic.net/20250926_241/1758878193109qM7E8_JPEG/Thumb_Poster.jpg?type=crop540_540' },
];
const REC_GENRE_LABEL = { action: 'Action', fantasy: 'Fantasy', comedy: 'Comedy', thriller: 'Thriller', romance: 'Romance' };
function recSeriesUrl(s) { return 'https://www.webtoons.com' + s.path + '/list?title_no=' + s.titleNo; }
// Webtoons accepts any slug in the viewer path when title_no/episode_no are valid
// (see buildWebtoonsChapterUrl); episode_no=1 is the series' first episode.
function recChapterUrl(s) { return 'https://www.webtoons.com' + s.path + '/x/viewer?title_no=' + s.titleNo + '&episode_no=1'; }

// Same per-browser id the reader uses (hm_bid) so the admin stats count one
// browser once across reader and start page. No account linkage.
function hmBid() {
  try {
    let v = localStorage.getItem('hm_bid');
    if (!v) { v = Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem('hm_bid', v); }
    return v;
  } catch { return 'anon'; }
}
function sendUsageBeacon(event, props) {
  try {
    const payload = JSON.stringify({ event, bid: hmBid(), props: props || {} });
    if (navigator.sendBeacon) navigator.sendBeacon('/api/beacon', new Blob([payload], { type: 'application/json' }));
    else fetch('/api/beacon', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload, keepalive: true });
  } catch { /* never let telemetry break the page */ }
}

(function renderRecommendedPicks() {
  const host = document.getElementById('picks-hero');
  const row = document.getElementById('picks-row');
  if (!host || !row || !urlRows) return;
  row.innerHTML = RECOMMENDED_SERIES.map(s =>
    '<button type="button" class="pick-tile" data-id="' + s.id + '">' +
      '<span class="pick-cover">' +
        '<img src="/api/proxy-image?url=' + encodeURIComponent(s.cover) + '" alt="" loading="lazy">' +
        '<span class="pick-genre" data-genre="' + s.genre + '">' + REC_GENRE_LABEL[s.genre] + '</span>' +
      '</span>' +
      '<span class="pick-zh" lang="zh-Hant">' + escapeHtml(s.zh) + '</span>' +
      '<span class="pick-en">' + escapeHtml(s.en) + '</span>' +
    '</button>').join('');
  host.hidden = false;
  row.querySelectorAll('.pick-tile').forEach(btn => btn.addEventListener('click', () => {
    const s = RECOMMENDED_SERIES.find(x => x.id === btn.dataset.id);
    if (!s) return;
    row.querySelectorAll('.pick-tile').forEach(b => b.classList.toggle('on', b === btn));
    urlRows.innerHTML = '';
    urlRows.appendChild(createUrlRow(recChapterUrl(s)));
    const chip = document.getElementById('pick-chip');
    if (chip) {
      chip.innerHTML = '<span><b lang="zh-Hant">' + escapeHtml(s.zh) + '</b> · ' + escapeHtml(s.en) +
        ' — chapter 1 is loaded. Pick your HSK level and press Read.</span>' +
        '<a href="' + recSeriesUrl(s) + '" target="_blank" rel="noopener">Official page ↗</a>';
      chip.hidden = false;
    }
    const form = document.getElementById('process-form');
    if (form) form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    sendUsageBeacon('rec_click', { id: s.id });
  }));
})();
