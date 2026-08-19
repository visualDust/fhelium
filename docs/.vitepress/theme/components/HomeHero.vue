<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<template>
  <section class="home-hero" aria-labelledby="home-title">
    <div class="home-hero-copy">
      <div class="home-brand-lockup">
        <span class="home-brand-name">
          <strong>FHElium</strong>
        </span>
        <span class="home-pronunciation-option"><b>“philium”</b><small>/ˈfɪliəm/</small></span>
        <span class="home-pronunciation-option"><b>“F-helium”</b><small>/ˌɛf ˈhiːliəm/</small></span>
      </div>
      <div class="home-hero-title">
        <h1 id="home-title">FHE, built from the tensor up.</h1>
        <span class="home-beta-bandage" aria-label="Beta development status"><span aria-hidden="true">Beta</span></span>
      </div>
      <p class="home-hero-summary">
        An open-source, cross-stack CKKS research framework with Python APIs, a typed PyTorch frontend, and native CPU and CUDA backends. Study program representation, cryptographic state, execution, runtime resources, and distributed systems independently or together through one inspectable value model.
      </p>
      <div class="home-hero-actions">
        <a class="is-primary" :href="withBase('/tutorial/')">Build &amp; run</a>
        <a :href="withBase('/tutorial/tutorials')">Browse tutorials</a>
      </div>
    </div>
    <div class="home-hero-visual" aria-hidden="true">
      <div class="home-hero-nucleus">
        <span class="home-hero-field is-proton is-upper" />
        <span class="home-hero-field is-neutron" />
        <span class="home-hero-field is-proton is-lower" />
      </div>
    </div>
  </section>
</template>
