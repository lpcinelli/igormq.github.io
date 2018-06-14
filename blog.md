---
layout: default
permalink: /blog
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

		<p class="date">{{ post.date | date: "%B %e, %Y" }}</p>
      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>
      <p class="entry">
        {{ post.excerpt }}
      </p>

      <!-- <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read more</a> -->
    </article>
  {% endfor %}
</div>
