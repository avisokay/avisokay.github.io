---
layout: page
permalink: /explainers/
title: Explainers
description: >
  I'm a visual learner. I love using whiteboards to think through problems, and wanted a space to
  share simplified explanations of things I think are really cool.
nav: true
nav_order: 3
---

<div class="row row-cols-1 row-cols-md-2 g-4 explainers">
  <div class="col">
    <div class="card h-100">
      <a href="/assets/explainers/ipd.html">
        {% include figure.liquid loading="eager" path="assets/img/durer_rhino.png" class="card-img-top" alt="Dürer's rhinoceros" %}
      </a>
      <div class="card-body">
        <h3 class="card-title"><a href="/assets/explainers/ipd.html">Inference on Predicted Data</a></h3>
        <p class="card-text">
          How do you perform valid inference when your dependent variable is imputed, or predicted,
          or otherwise not measured directly?
        </p>
        <p class="card-text">
          <a href="/assets/explainers/ipd_japanese.html">日本語版</a>
        </p>
      </div>
    </div>
  </div>
  <div class="col">
    <div class="card h-100">
      <a href="https://avisokay.shinyapps.io/uw_ard_viz/" target="_blank" rel="noopener">
        {% include figure.liquid loading="eager" path="assets/img/publication_preview/ard.png" class="card-img-top" alt="Aggregated relational data visualization" %}
      </a>
      <div class="card-body">
        <h3 class="card-title">
          <a href="https://avisokay.shinyapps.io/uw_ard_viz/" target="_blank" rel="noopener">
            Can You Estimate Network Characteristics Without Observing the Full Network?
          </a>
        </h3>
        <p class="card-text">
          An introduction to Aggregated Relational Data (ARD) — <em>How many people with trait X do
          you know?</em>
        </p>
      </div>
    </div>
  </div>
</div>
