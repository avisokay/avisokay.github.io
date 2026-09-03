---
layout: page
permalink: /research/
title: Research
description: Publications in reverse chronological order.
nav: true
nav_order: 2
---

<!-- Sections come from --query filters on the local `status` field in
     _bibliography/papers.bib. _config.yml sets scholar.group_by to `none` so
     jekyll-scholar does not additionally re-split each section by year. -->

<div class="publications">

<h2 class="bibliography">Peer Reviewed Publications <span class="heading-note">(* denotes co-first-author)</span></h2>
{% bibliography --query @*[status=published] %}

<h2 class="bibliography">Working Papers</h2>
{% bibliography --query @*[status=working] %}

<h2 class="bibliography">Other Writings</h2>
{% bibliography --query @*[status=other] %}

</div>
