---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

<div id="about-page">
  <p class="about-lead">LLM과 retrieval/search 분야 논문을 읽고 정리하는 기록용 블로그입니다.</p>

  <dl class="about-profile">
    <dt>이름</dt>
    <dd>이석기 <span class="about-sub">Seokgi Lee</span></dd>
    <dt>이메일</dt>
    <dd><a href="mailto:seokilee@snu.ac.kr">seokilee@snu.ac.kr</a></dd>
    <dt>GitHub</dt>
    <dd><a href="https://github.com/seokilee0412">github.com/seokilee0412</a></dd>
    <dt>관심 도메인</dt>
    <dd>LLM, retrieval/search</dd>
  </dl>

  <h2 class="about-head">이 블로그에서는</h2>
  <p class="about-text">2025년 8월부터 시간 날때마다 틈틈히 논문 리뷰를 진행하고 있습니다.</p>

  <ul class="about-topics">
    {% for t in site.data.topics %}
      {% assign t_count = 0 %}
      {% for post in site.posts %}
        {% include topic-of.html post=post %}
        {% if topic and topic.id == t.id %}{% assign t_count = t_count | plus: 1 %}{% endif %}
      {% endfor %}
      {% if t_count > 0 %}
        <li>
          <a href="{{ '/topics/' | relative_url }}#{{ t.id }}">{{ t.name }}</a>
          <span class="tag-leader" aria-hidden="true"></span>
          <span class="tag-count">{{ t_count }}편</span>
        </li>
      {% endif %}
    {% endfor %}
  </ul>

  <h2 class="about-head">최근 리뷰</h2>
  <ul class="about-recent">
    {% for post in site.posts limit: 3 %}
      <li>
        <a href="{{ post.url | relative_url }}" title="{{ post.title | escape }}">{{ post.title | split: ':' | first | strip }}</a>
        <span class="tag-leader" aria-hidden="true"></span>
        <time class="tag-count">{{ post.date | date: '%y.%m.%d' }}</time>
      </li>
    {% endfor %}
  </ul>
</div>
