/**
 * Bilibili Manga (哔哩哔哩漫画) adapter — NOT SUPPORTED, throws a clear error.
 *
 * Everything below was measured on 2026-08-31. Two earlier versions of this
 * comment stated guesses in the same voice as facts and sent people down the
 * wrong path twice, so where something is inferred it now says so.
 *
 * WHAT WAS TESTED
 *
 * Plain server-side POST, no browser: every endpoint returns
 * `{ code: 99, msg: "请求失败,请稍后重试。" }` — including ComicDetail, which is
 * public metadata needing no account. So code 99 means "request lacks the
 * session state a browser picks up on first load", NOT "not logged in".
 *
 * Inside headless Chromium, NOT logged in:
 *   free chapter -> GetImageIndex code=0, ImageToken code=0 (x4)
 *   paid chapter -> GetImageIndex code=1, ImageToken never called
 * The only 401s are account features (GetEpisodeBuyInfo, AddHistory,
 * HasFavorite), irrelevant to reading. code=1 is a clean paywall signal if
 * anyone ever needs to tell "locked" from "broken".
 *
 * WHY IT STILL DOESN'T WORK
 *
 * Those responses carry an empty `data` field — the real payload is in
 * `bytesData`, an encoded blob (~9KB for GetImageIndex). Image URLs can't be
 * read from the responses. They can't be observed indirectly either: in a
 * headless session the artwork never loads at all — zero manga images in the
 * DOM, and the manga CDN serves only ~39KB of incidental files. There is
 * nothing to harvest. The remaining path is reverse-engineering the bytesData
 * encoding, which is a deliberate protection measure and would break on any
 * format change.
 *
 * WHY IT WOULDN'T BE WORTH IT ANYWAY
 *
 * Bilibili Manga sells chapters. Each series has a publisher-set free preview;
 * past it costs 漫币 or a 漫读券 voucher (79 漫币 on 蓝色监狱 ch.14 — price and
 * preview length both vary by title). 限时免费 and 等就免费 further scope access
 * to a particular ACCOUNT at a particular TIME. Per search.ts, sources that die
 * partway through a series stay out of onsite search regardless, so this would
 * serve only pasted URLs and fail unpredictably.
 *
 * CORRECTIONS TO EARLIER VERSIONS
 *
 * "Requires SESSDATA cookie authentication" — wrong; free chapters return
 * code=0 with no account. "A shared service account violates ToS, banned
 * within days" — no evidence was ever gathered; removed rather than repeated.
 * An account would not have helped either way, since the blocker is payload
 * encoding, not identity.
 *
 * matches() still returns true so the dispatcher catches Bilibili URLs and the
 * user sees a clear error rather than the generic "no images found" from the
 * Puppeteer fallback — which would also fail here, since it scrapes <img> tags
 * and there are none.
 */
import { SiteAdapter, ChapterImages } from './types';

const CHAPTER_RE = /^https?:\/\/manga\.bilibili\.com\/(?:m\/)?mc(\d+)\/(\d+)/;

export const bilibiliMangaAdapter: SiteAdapter = {
  name: 'Bilibili Manga',
  sourceId: 'bilibili-manga',
  matches(url) {
    return CHAPTER_RE.test(url);
  },
  async getChapter(_url): Promise<ChapterImages> {
    const err: any = new Error('Bilibili Manga chapters can\'t be fetched. Paid chapters need a purchase, and even free ones are only readable inside Bilibili\'s own reader. Try uploading pages you already have, or a chapter link from a site that serves its images publicly.');
    err.code = 'UNSUPPORTED_SITE';
    throw err;
  },
};
