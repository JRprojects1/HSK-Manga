/* ============================================================
   Account widget — Recent drawer + sign-in modal + auth pill.
   Self-contained: injects its own DOM, talks to the same backend
   endpoints as the app page (/api/me, /api/auth/*), and reads the
   same localStorage `jobHistory` the app page writes. Safe to load
   on any page — it no-ops if its nav mount points are absent.

   Mount points the host page must provide:
     #aw-recent-btn   — a <button> in the nav (opens the drawer)
     #aw-auth-pill    — an empty container in the nav (filled here)

   NOTE: This deliberately parallels the drawer/auth logic in app.js
   for now. Item 2 (accounts rethink) is the point to unify them.
   ============================================================ */
(function () {
  'use strict';

  var currentUser = null;

  // ---- helpers ----
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function shortenUrl(u) { u = String(u || ''); return u.length > 46 ? u.slice(0, 43) + '…' : u; }
  function hskLabel(level) { var n = String(level || '').replace(/\D/g, ''); return n ? 'HSK ' + n : ''; }
  function relativeTime(ts) {
    if (!ts) return '';
    var s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (s < 60) return 'just now';
    var m = Math.floor(s / 60); if (m < 60) return m + ' min ago';
    var h = Math.floor(m / 60); if (h < 24) return h + (h === 1 ? ' hour ago' : ' hours ago');
    var d = Math.floor(h / 24); if (d < 7) return d + (d === 1 ? ' day ago' : ' days ago');
    var w = Math.floor(d / 7); if (w < 5) return w + (w === 1 ? ' week ago' : ' weeks ago');
    return new Date(ts).toLocaleDateString();
  }
  function loadHistory() {
    try { var h = JSON.parse(localStorage.getItem('jobHistory') || '[]'); return Array.isArray(h) ? h : []; }
    catch (e) { return []; }
  }

  var BOOK = '<svg viewBox="0 0 24 24"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>';
  var CHEVRON = '<svg viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>';

  // ---- injected DOM ----
  function injectDom() {
    var el = document.createElement('div');
    el.innerHTML = [
      '<div class="aw-backdrop aw-v" id="aw-backdrop"></div>',
      '<aside class="aw-drawer aw-v" id="aw-drawer" aria-label="My chapters">',
      '  <div class="aw-drawer-head">',
      '    <h2>My chapters</h2>',
      '    <button class="aw-icon-btn" id="aw-drawer-close" type="button" aria-label="Close">',
      '      <svg viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>',
      '    </button>',
      '  </div>',
      '  <div class="aw-drawer-body" id="aw-recent-list"></div>',
      '</aside>',
      '<div class="aw-modal-backdrop aw-v aw-hidden" id="aw-signin-modal">',
      '  <div class="aw-modal">',
      '    <div class="aw-modal-icon"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>',
      '    <div id="aw-step-primary">',
      '      <h2 class="aw-modal-title">Sign in to HSK Manga</h2>',
      '      <p class="aw-modal-sub">No account? Your chapters stay in your browser. Sign in to read them on any device — guest use is still fine.</p>',
      '      <button id="aw-google-btn" class="aw-btn-google" type="button">',
      '        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">',
      '          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.56c2.08-1.92 3.28-4.74 3.28-8.1z"/>',
      '          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.56-2.77c-.99.66-2.25 1.05-3.72 1.05-2.87 0-5.29-1.94-6.16-4.54H2.18v2.85A11 11 0 0 0 12 23z"/>',
      '          <path fill="#FBBC05" d="M5.84 14.08a6.6 6.6 0 0 1 0-4.16V7.07H2.18a11 11 0 0 0 0 9.86l3.66-2.85z"/>',
      '          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15A10.55 10.55 0 0 0 12 1a11 11 0 0 0-9.82 6.07l3.66 2.85c.87-2.6 3.3-4.54 6.16-4.54z"/>',
      '        </svg>Continue with Google',
      '      </button>',
      '      <div class="aw-divider"><span>or</span></div>',
      '      <button id="aw-email-toggle" class="aw-btn-secondary" type="button">Sign in with a code sent to your email</button>',
      '      <button id="aw-cancel-primary" class="aw-cancel" type="button">Continue as guest</button>',
      '    </div>',
      '    <div id="aw-step-email" class="aw-hidden">',
      '      <h2 class="aw-modal-title">Sign in with email</h2>',
      '      <p class="aw-modal-sub">We\'ll send a 6-digit code to your email. No password needed.</p>',
      '      <input type="email" id="aw-email" class="aw-input" placeholder="you@example.com" autocomplete="email" inputmode="email">',
      '      <div id="aw-error-1" class="aw-error aw-hidden"></div>',
      '      <button id="aw-send-code" class="aw-btn-primary" type="button">Send code</button>',
      '      <button id="aw-back-primary" class="aw-btn-secondary" type="button">Back</button>',
      '    </div>',
      '    <div id="aw-step-code" class="aw-hidden">',
      '      <h2 class="aw-modal-title">Check your email</h2>',
      '      <p class="aw-modal-sub">We sent a 6-digit code to <span id="aw-email-echo" class="aw-email-echo"></span>. Enter it below.</p>',
      '      <input type="text" id="aw-code" class="aw-input" placeholder="123456" inputmode="numeric" maxlength="6" autocomplete="one-time-code">',
      '      <div id="aw-error-2" class="aw-error aw-hidden"></div>',
      '      <button id="aw-verify-code" class="aw-btn-primary" type="button">Verify &amp; sign in</button>',
      '      <button id="aw-back-email" class="aw-btn-secondary" type="button">Use a different email</button>',
      '    </div>',
      '  </div>',
      '</div>'
    ].join('');
    while (el.firstChild) document.body.appendChild(el.firstChild);
  }

  var $;
  function byId(id) { return document.getElementById(id); }

  // ---- My chapters (drawer) ----
  // Signed in → account library (GET /api/library); signed out → the anonymous
  // device bucket (localStorage). They stay separate; chapters cross over only
  // via an explicit claim (the banner). Same model as the app page.
  function renderRecent() {
    var list = byId('aw-recent-list');
    if (!list) return;
    if (currentUser && currentUser.email) {
      fetch('/api/library', { credentials: 'same-origin' })
        .then(function (r) { return r.json().catch(function () { return {}; }); })
        .then(function (body) {
          if (body && body.signedIn === false) {
            // Session expired server-side — reconcile to signed-out, show the bucket.
            currentUser = null; renderPill(); renderList(list, null); return;
          }
          if (body && body.signedIn && Array.isArray(body.library)) {
            renderList(list, body.library.map(function (e) {
              return { url: e.url, resultUrl: e.resultUrl, title: e.title, resolved: e.resolved, level: e.level, timestamp: e.addedAt };
            }));
          } else {
            renderLibraryError(list); // don't blank the account / mislabel bucket as unclaimed
          }
        })
        .catch(function () { renderLibraryError(list); });
    } else {
      renderList(list, null); // signed out → browser bucket only
    }
  }
  function renderLibraryError(list) {
    list.innerHTML = '<p class="aw-drawer-empty">Couldn’t load your chapters.<br><button type="button" id="aw-library-retry" style="margin-top:10px;background:none;border:none;color:var(--aw-accent);font:inherit;font-weight:700;cursor:pointer;text-decoration:underline">Retry</button></p>';
    var rb = byId('aw-library-retry');
    if (rb) rb.addEventListener('click', renderRecent);
  }

  function rowHtml(item) {
    var hasReader = !!item.resultUrl;
    var href = hasReader ? esc(item.resultUrl) : ('/start?url=' + encodeURIComponent(item.url));
    var target = hasReader ? '_blank' : '_self';
    var sub = hasReader ? relativeTime(item.timestamp || Date.now()) : 'Click to load';
    var badge = item.level ? '<span class="aw-recent-level">' + esc(hskLabel(item.level)) + '</span>' : '';

    // Learner-friendly resolution, when present. Title = English, else the
    // Chinese series name (never a fabricated name); cover replaces the book
    // icon; a hanzi + pinyin line and the normalized chapter number ride below.
    var r = item.resolved;
    var titleText = (r && (r.english || r.seriesZh)) || item.title || shortenUrl(item.url);
    var thumb = (r && r.coverUrl)
      ? '<span class="aw-recent-cover"><img src="' + esc(r.coverUrl) + '" alt="" loading="lazy"></span>'
      : '<span class="aw-recent-thumb">' + BOOK + '</span>';
    // The Chinese + pinyin aid line only applies to actual Han series names; an
    // English title never gets one (its "pinyin" would just echo the English).
    var hasHan = !!(r && r.seriesZh && /[一-鿿]/.test(r.seriesZh));
    var cnLine = '';
    if (hasHan && r.english) {
      cnLine = '<span class="aw-recent-cn">' + esc(r.seriesZh) + (r.seriesPinyin ? ' · <span class="aw-recent-py">' + esc(r.seriesPinyin) + '</span>' : '') + '</span>';
    } else if (hasHan && r.seriesPinyin) {
      cnLine = '<span class="aw-recent-cn"><span class="aw-recent-py">' + esc(r.seriesPinyin) + '</span></span>';
    }
    var chap = (r && r.chapterLabel) ? '<span class="aw-recent-chap">' + esc(r.chapterLabel) + '</span>' : '';

    return '<div class="aw-recent-item-wrap" data-url="' + esc(item.url) + '" data-level="' + esc(item.level || '') + '" style="display:flex;align-items:center;gap:6px">' +
      '<a class="aw-recent-item" href="' + href + '" target="' + target + '" style="flex:1">' +
      thumb +
      '<span class="aw-recent-meta">' +
      '<span class="aw-recent-title">' + esc(titleText) + '</span>' +
      cnLine +
      '<span class="aw-recent-sub">' + badge + chap + esc(sub) + '</span>' +
      '</span>' +
      '<span class="aw-recent-thumb" style="background:none;color:var(--aw-muted);width:auto;height:auto">' + CHEVRON + '</span>' +
      '</a>' +
      '<button class="aw-recent-del" type="button" aria-label="Remove">×</button>' +
      '</div>';
  }

  function renderList(list, accountItems) {
    var anon = loadHistory();
    var banner = '', status = '', items;
    if (currentUser && currentUser.email) {
      var accKeys = {};
      (accountItems || []).forEach(function (i) { accKeys[i.url + '|' + (i.level || 'hsk3')] = 1; });
      var anonNew = anon.filter(function (h) { return !accKeys[h.url + '|' + (h.level || 'hsk3')]; });
      items = anonNew.concat(accountItems || []); // unclaimed browser chapters first, then synced
      // Only http URLs are claimable (image uploads carry a non-http placeholder);
      // gate the banner on those but still show upload rows.
      var claimable = anonNew.filter(function (h) { return h && typeof h.url === 'string' && h.url.indexOf('http') === 0; });
      if (claimable.length > 0) {
        var n = claimable.length;
        banner = '<div class="aw-claim-banner"><span>' + n + ' chapter' + (n === 1 ? '' : 's') +
          ' in your browser ' + (n === 1 ? "isn't" : "aren't") + ' in your account yet.</span>' +
          '<button type="button" id="aw-claim-btn">Add to my account</button></div>';
      } else if (items.length) {
        status = '<p class="aw-drawer-status">Synced to your account · ' + esc(currentUser.email) + '</p>';
      }
    } else {
      items = anon;
      if (items.length) status = '<p class="aw-drawer-status">Saved to your browser. Sign in to read your chapters on any device.</p>';
    }
    if (!items.length && !banner) {
      var hint = (currentUser && currentUser.email) ? '' : '<span style="display:block;margin-top:10px;font-size:12px;font-weight:600;opacity:.8">Sign in to see any chapters saved to your account.</span>';
      list.innerHTML = '<p class="aw-drawer-empty">No chapters yet' + hint + '</p>';
      return;
    }
    list.innerHTML = banner + status + items.map(rowHtml).join('');
    var cb = byId('aw-claim-btn');
    if (cb) cb.addEventListener('click', claimAnonymous);
  }

  function claimAnonymous() {
    var entries = loadHistory()
      .filter(function (h) { return h && typeof h.url === 'string' && h.url.indexOf('http') === 0; })
      .slice(0, 500)
      .map(function (h) { return { url: h.url, resultUrl: h.resultUrl || undefined, title: h.title || '', level: h.level || 'hsk3', lang: h.lang || 'auto' }; });
    if (!entries.length) return;
    var claimedKeys = {};
    entries.forEach(function (e) { claimedKeys[e.url + '|' + e.level] = 1; });
    fetch('/api/library/claim', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ entries: entries })
    }).then(function (r) {
      if (r.ok) {
        // Clear only what we sent (preserve non-http/upload chapters).
        var survivors = loadHistory().filter(function (h) { return !claimedKeys[h.url + '|' + (h.level || 'hsk3')]; });
        localStorage.setItem('jobHistory', JSON.stringify(survivors));
        renderRecent();
      }
    }).catch(function () {});
  }
  function onListClick(e) {
    // Opening a chapter → stamp lastReadAt on the account entry (fire-and-forget;
    // server no-ops for guests). Mirrors app.js — keeps "chapters read" measurable.
    var openLink = e.target.closest ? e.target.closest('.aw-recent-item') : null;
    if (openLink && currentUser && currentUser.email) {
      var openWrap = openLink.closest('.aw-recent-item-wrap');
      var openUrl = openWrap && openWrap.getAttribute('data-url');
      if (openUrl) {
        try {
          fetch('/api/library/touch', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: openUrl, level: openWrap.getAttribute('data-level') || undefined }),
            keepalive: true,
          });
        } catch (err) {}
      }
    }
    var del = e.target.closest ? e.target.closest('.aw-recent-del') : null;
    if (!del) return;
    e.preventDefault();
    var wrap = del.closest('.aw-recent-item-wrap');
    var url = wrap && wrap.getAttribute('data-url');
    if (!url) return;
    var level = (wrap && wrap.getAttribute('data-level')) || undefined;
    if (!window.confirm('Remove this chapter from your list?')) return;
    // Route by ROW: always drop the bucket copy; only hit the account endpoint
    // with a concrete level (blank level = bucket-only row; an omitted level
    // would wipe every level of that url on the server).
    var anon = loadHistory().filter(function (h) { return !(h.url === url && (h.level || 'hsk3') === (level || 'hsk3')); });
    localStorage.setItem('jobHistory', JSON.stringify(anon));
    if (currentUser && currentUser.email && level) {
      fetch('/api/library/remove', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin', body: JSON.stringify({ url: url, level: level })
      }).catch(function () {}).then(function () { renderRecent(); });
    } else {
      renderRecent();
    }
  }
  function openDrawer() {
    renderRecent();
    var d = byId('aw-drawer'), b = byId('aw-backdrop');
    if (d) d.classList.add('aw-open');
    if (b) b.classList.add('aw-open');
  }
  function closeDrawer() {
    var d = byId('aw-drawer'), b = byId('aw-backdrop');
    if (d) d.classList.remove('aw-open');
    if (b) b.classList.remove('aw-open');
  }

  // ---- Auth pill ----
  function renderPill() {
    var pill = byId('aw-auth-pill');
    if (!pill) return;
    if (currentUser && currentUser.email) {
      pill.innerHTML =
        '<button class="aw-user-btn" id="aw-user-btn" type="button">' +
        '<span class="aw-user-email">' + esc(currentUser.email) + '</span>' +
        '<svg viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg></button>' +
        '<div class="aw-user-dropdown aw-hidden" id="aw-user-dropdown"><button id="aw-signout" type="button">Sign out</button></div>';
      byId('aw-user-btn').addEventListener('click', function (e) {
        e.stopPropagation();
        byId('aw-user-dropdown').classList.toggle('aw-hidden');
      });
      byId('aw-signout').addEventListener('click', signOut);
    } else {
      pill.innerHTML = '<button class="aw-signin-btn" id="aw-signin-btn" type="button">Sign in</button>';
      byId('aw-signin-btn').addEventListener('click', openModal);
    }
  }
  function fetchMe() {
    fetch('/api/me', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) { currentUser = (data && data.user) || null; renderPill(); maybeWelcome(); })
      .catch(function () { /* leave as guest */ });
  }
  // Strip the one-time Google sign-in flag from the URL (cosmetic). We do NOT
  // auto-open the drawer on the front page — the claim nudge lives on /start, so
  // the marketing page never pops a panel open on sign-in.
  function maybeWelcome() {
    try {
      var p = new URLSearchParams(location.search);
      if (p.get('signedin') === '1') {
        p.delete('signedin');
        var qs = p.toString();
        history.replaceState({}, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
      }
    } catch (e) {}
  }
  function signOut() {
    fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' })
      .catch(function () {})
      .then(function () { currentUser = null; renderPill(); renderRecent(); });
  }

  // ---- Sign-in modal ----
  function showStep(id) {
    ['aw-step-primary', 'aw-step-email', 'aw-step-code'].forEach(function (s) {
      var el = byId(s); if (el) el.classList.toggle('aw-hidden', s !== id);
    });
    ['aw-error-1', 'aw-error-2'].forEach(function (s) { var el = byId(s); if (el) el.classList.add('aw-hidden'); });
  }
  function openModal() { showStep('aw-step-primary'); byId('aw-signin-modal').classList.remove('aw-hidden'); }
  function closeModal() { byId('aw-signin-modal').classList.add('aw-hidden'); }
  function showErr(id, msg) { var el = byId(id); if (el) { el.textContent = msg; el.classList.remove('aw-hidden'); } }

  function sendCode() {
    var email = (byId('aw-email').value || '').trim();
    if (!email) { showErr('aw-error-1', 'Please enter your email.'); return; }
    var btn = byId('aw-send-code'); btn.disabled = true; btn.textContent = 'Sending…';
    fetch('/api/auth/magic-link/request', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ email: email })
    }).then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok) { showErr('aw-error-1', res.d.error || 'Could not send the code.'); return; }
        byId('aw-email-echo').textContent = email;
        showStep('aw-step-code');
        setTimeout(function () { byId('aw-code').focus(); }, 20);
      })
      .catch(function () { showErr('aw-error-1', 'Network error. Please try again.'); })
      .then(function () { btn.disabled = false; btn.textContent = 'Send code'; });
  }
  function verifyCode() {
    var email = (byId('aw-email').value || '').trim();
    var code = (byId('aw-code').value || '').trim();
    if (!code) { showErr('aw-error-2', 'Please enter the code.'); return; }
    var btn = byId('aw-verify-code'); btn.disabled = true; btn.textContent = 'Verifying…';
    fetch('/api/auth/magic-link/verify', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin', body: JSON.stringify({ email: email, code: code })
    }).then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok || !res.d.user) { showErr('aw-error-2', res.d.error || 'Invalid code.'); return; }
        currentUser = res.d.user; renderPill(); closeModal(); renderRecent();
      })
      .catch(function () { showErr('aw-error-2', 'Network error. Please try again.'); })
      .then(function () { btn.disabled = false; btn.textContent = 'Verify & sign in'; });
  }

  // ---- wire up ----
  function init() {
    var recentBtn = byId('aw-recent-btn');
    var pill = byId('aw-auth-pill');
    if (!recentBtn && !pill) return; // host page didn't opt in

    injectDom();

    if (recentBtn) recentBtn.addEventListener('click', openDrawer);
    var dc = byId('aw-drawer-close'); if (dc) dc.addEventListener('click', closeDrawer);
    var bd = byId('aw-backdrop'); if (bd) bd.addEventListener('click', closeDrawer);
    var listEl = byId('aw-recent-list'); if (listEl) listEl.addEventListener('click', onListClick);

    byId('aw-google-btn').addEventListener('click', function () { window.location.href = '/api/auth/google?return=' + encodeURIComponent(location.pathname); });
    byId('aw-email-toggle').addEventListener('click', function () { showStep('aw-step-email'); setTimeout(function () { byId('aw-email').focus(); }, 20); });
    byId('aw-cancel-primary').addEventListener('click', closeModal);
    byId('aw-back-primary').addEventListener('click', function () { showStep('aw-step-primary'); });
    byId('aw-back-email').addEventListener('click', function () { showStep('aw-step-email'); });
    byId('aw-send-code').addEventListener('click', sendCode);
    byId('aw-verify-code').addEventListener('click', verifyCode);
    byId('aw-email').addEventListener('keydown', function (e) { if (e.key === 'Enter') sendCode(); });
    byId('aw-code').addEventListener('keydown', function (e) { if (e.key === 'Enter') verifyCode(); });

    byId('aw-signin-modal').addEventListener('click', function (e) { if (e.target === this) closeModal(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { closeDrawer(); closeModal(); } });
    document.addEventListener('click', function () { var dd = byId('aw-user-dropdown'); if (dd) dd.classList.add('aw-hidden'); });

    if (pill) fetchMe();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
