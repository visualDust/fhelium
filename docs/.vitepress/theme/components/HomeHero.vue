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
        FHElium gives both usage models access to the same CKKS stack—from public values and mixed-level IR to Backend linking, RNS and NTT, native CPU/CUDA arithmetic, and rank-local execution.
      </p>
      <div class="home-hero-actions">
        <a class="is-primary" :href="withBase('/tutorial/')">Get started</a>
        <a :href="withBase('/concepts/')">Explore the stack</a>
        <a
          class="is-github"
          href="https://github.com/VisualDust/fhelium"
          target="_blank"
          rel="noopener noreferrer"
        >
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="M12 .7a11.3 11.3 0 0 0-3.57 22c.57.1.78-.25.78-.55v-2.17c-3.18.7-3.85-1.36-3.85-1.36-.52-1.33-1.27-1.68-1.27-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.75 1.18 1.75 1.18 1.02 1.75 2.68 1.25 3.33.96.1-.74.4-1.25.73-1.54-2.54-.29-5.21-1.27-5.21-5.59 0-1.24.44-2.25 1.18-3.04-.12-.29-.51-1.46.11-3 0 0 .96-.31 3.11 1.16a10.8 10.8 0 0 1 5.68 0c2.16-1.47 3.11-1.16 3.11-1.16.62 1.54.23 2.71.11 3 .74.79 1.18 1.8 1.18 3.04 0 4.33-2.68 5.29-5.23 5.58.41.36.78 1.06.78 2.14v3.18c0 .3.21.66.79.55A11.3 11.3 0 0 0 12 .7Z" />
          </svg>
          GitHub
        </a>
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
