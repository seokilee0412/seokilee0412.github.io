/*
 * 사이트 보조 스크립트 (테마 번들과 별개로 동작)
 *  1) 홈: 주제 필터 칩으로 카드 걸러보기 (#topic=<id> 로 공유 가능)
 *  2) 포스트: ①②③… 로 시작하는 소제목에 별도 클래스를 붙여 위계를 구분
 *  3) TOPICS 탭: 주제 × 월 타임라인을 그리고, 접이식 목차와 연결
 * 글 내용은 바꾸지 않고 표시 방식만 다룹니다.
 */
(function () {
  'use strict';

  function initTopicFilter() {
    var bar = document.getElementById('topic-filter');
    var list = document.getElementById('post-list');
    if (!bar || !list) return;

    var chips = Array.prototype.slice.call(bar.querySelectorAll('.topic-chip'));
    var cards = Array.prototype.slice.call(list.querySelectorAll('article[data-topic]'));
    var known = chips.map(function (c) { return c.getAttribute('data-topic'); });

    function apply(id, push) {
      if (known.indexOf(id) === -1) id = 'all';
      chips.forEach(function (c) {
        var on = c.getAttribute('data-topic') === id;
        c.classList.toggle('active', on);
        c.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      cards.forEach(function (card) {
        var show = id === 'all' || card.getAttribute('data-topic') === id;
        card.classList.toggle('topic-hidden', !show);
      });
      if (push) {
        var hash = id === 'all' ? '' : '#topic=' + id;
        try {
          history.replaceState(null, '', location.pathname + location.search + hash);
        } catch (e) { /* ignore */ }
      }
    }

    chips.forEach(function (c) {
      c.addEventListener('click', function () {
        apply(c.getAttribute('data-topic'), true);
      });
    });

    var m = /topic=([\w-]+)/.exec(location.hash);
    if (m) apply(m[1], false);
  }

  function markSubHeadings() {
    var content = document.querySelector('article[data-toc] .content');
    if (!content) return;
    var circled = /^[①-⑳❶-❿]/; // ①-⑳, ❶-❿
    content.querySelectorAll('h4, h5').forEach(function (h) {
      if (circled.test((h.textContent || '').trim())) h.classList.add('step-heading');
    });

    // 제목 앞 번호(1. / 1.2. / A. / ①)를 옅은 색으로 분리
    var mark = /^(\s*)((?:\d+\.)+\d*|[A-Z]\.|[\u2460-\u2473\u2776-\u277F])(?=\s|$)(\s*)/;
    content.querySelectorAll('h1, h2, h3, h4, h5').forEach(function (h) {
      var walker = document.createTreeWalker(h, NodeFilter.SHOW_TEXT, null);
      var node = walker.nextNode();
      while (node && !node.nodeValue.trim()) node = walker.nextNode();
      if (!node) return;
      var m = mark.exec(node.nodeValue);
      if (!m) return;
      var span = document.createElement('span');
      span.className = 'hd-mark';
      span.textContent = m[2];
      var rest = node.nodeValue.slice(m[0].length);
      node.parentNode.insertBefore(span, node);
      node.nodeValue = rest;
    });

    // 첫 문단의 "Written by. …" 저자 표기
    var first = content.firstElementChild;
    if (first && first.tagName === 'P' && /^\s*written\s+by/i.test(first.textContent || '')) {
      first.classList.add('byline');
    }
  }

  /* ---------- TOPICS 탭: 타임라인 + 접이식 목차 ---------- */
  function initTopicsPage() {
    var mount = document.getElementById('topic-timeline');
    var src = document.getElementById('topic-timeline-data');
    if (!mount || !src) return;

    var data;
    try { data = JSON.parse(src.textContent); } catch (e) { return; }
    var posts = data.posts.filter(function (p) { return p.topic; });
    if (!posts.length) return;

    // 첫 글이 있는 달 ~ 이번 달
    var ym = function (d) { return { y: +d.slice(0, 4), m: +d.slice(5, 7) }; };
    var dates = posts.map(function (p) { return p.date; }).sort();
    var start = ym(dates[0]);
    var now = new Date();
    var endY = now.getFullYear(), endM = now.getMonth() + 1;
    var last = ym(dates[dates.length - 1]);
    if (last.y * 12 + last.m > endY * 12 + endM) { endY = last.y; endM = last.m; }

    var months = [];
    for (var y = start.y, m = start.m; y * 12 + m <= endY * 12 + endM; m++) {
      if (m > 12) { m = 1; y++; }
      months.push({ y: y, m: m });
    }

    function el(tag, cls, html) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (html != null) e.innerHTML = html;
      return e;
    }
    function esc(s) {
      return String(s).replace(/[&<>"]/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
      });
    }

    var tl = el('div', 'tl');
    tl.style.setProperty('--months', months.length);

    // 머리줄: 연도 + 월
    var head = el('div', 'tl-row tl-head');
    head.appendChild(el('div'));
    months.forEach(function (mo, i) {
      var yr = i === 0 || mo.m === 1 ? '<b>' + mo.y + '</b>' : '&nbsp;';
      head.appendChild(el('div', null, yr + '<br>' + mo.m + '월'));
    });
    tl.appendChild(head);

    var body = el('div', 'tl-body');
    var rows = {};
    data.topics.forEach(function (t) {
      var ps = posts.filter(function (p) { return p.topic === t.id; });
      if (!ps.length) return;
      var row = el('div', 'tl-row');
      row.setAttribute('role', 'button');
      row.setAttribute('tabindex', '0');
      row.setAttribute('data-topic', t.id);
      row.setAttribute('aria-label', t.name + ' ' + ps.length + '편, 목록 펼치기');
      row.appendChild(el('div', 'tl-label', esc(t.name)));
      months.forEach(function (mo) {
        var cell = el('div', 'tl-cell' + (mo.m === 1 ? ' year' : ''));
        ps.forEach(function (p) {
          var d = ym(p.date);
          if (d.y === mo.y && d.m === mo.m) {
            var a = el('a', 'tl-dot');
            a.href = p.url;
            a.title = p.title + ' (' + p.date.replace(/-/g, '.') + ')';
            a.setAttribute('aria-label', p.title);
            a.addEventListener('click', function (ev) { ev.stopPropagation(); });
            cell.appendChild(a);
          }
        });
        row.appendChild(cell);
      });
      body.appendChild(row);
      rows[t.id] = row;
    });
    tl.appendChild(body);

    var scroll = el('div', 'tl-scroll');
    scroll.appendChild(tl);
    mount.appendChild(scroll);
    // 화면이 좁으면 최근 달이 먼저 보이도록
    if (scroll.scrollWidth > scroll.clientWidth) scroll.scrollLeft = scroll.scrollWidth;
    mount.appendChild(el('p', 'tl-hint', '점 하나가 논문 한 편이에요. 점에 마우스를 올리면 제목이 보이고, 주제를 누르면 아래 목록이 펼쳐집니다.'));

    function highlight(id) {
      Object.keys(rows).forEach(function (k) { rows[k].classList.toggle('on', k === id); });
    }

    function openGroup(id, scrollTo) {
      var g = document.getElementById(id);
      if (!g || g.tagName !== 'DETAILS') return;
      g.open = true;
      highlight(id);
      if (scrollTo) g.scrollIntoView({ behavior: 'smooth', block: 'start' });
      try { history.replaceState(null, '', '#' + id); } catch (e) { /* ignore */ }
    }

    Object.keys(rows).forEach(function (id) {
      var row = rows[id];
      row.addEventListener('click', function () { openGroup(id, true); });
      row.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openGroup(id, true); }
      });
    });

    // 목차를 직접 펼치거나 접을 때 타임라인 강조 동기화
    document.querySelectorAll('details.toc-group').forEach(function (g) {
      g.addEventListener('toggle', function () {
        if (g.open) highlight(g.id);
        else if (rows[g.id] && rows[g.id].classList.contains('on')) highlight(null);
      });
    });

    // /topics/#<id> 로 들어오면 해당 주제를 펼침
    var hash = decodeURIComponent(location.hash.slice(1));
    if (hash) openGroup(hash, true);
  }

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    initTopicFilter();
    markSubHeadings();
    initTopicsPage();
  });
})();
