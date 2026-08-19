import { inject as injectAnalytics } from '@vercel/analytics'
import { injectSpeedInsights } from '@vercel/speed-insights'
import DefaultTheme from 'vitepress/theme'
import { inBrowser, type Router, type Theme } from 'vitepress'
import { defineAsyncComponent } from 'vue'

import BlogIndex from './components/BlogIndex.vue'
import BsgsMatvecPerformance from './components/BsgsMatvecPerformance.vue'
import DocCard from './components/DocCard.vue'
import DocGrid from './components/DocGrid.vue'
import HomeControlDeck from './components/HomeControlDeck.vue'
import HomeGraphXray from './components/HomeGraphXray.vue'
import HomeHero from './components/HomeHero.vue'
import HomeOpeningRail from './components/HomeOpeningRail.vue'
import HomeStackBuilder from './components/HomeStackBuilder.vue'
import InstallCommand from './components/InstallCommand.vue'
import MermaidDiagram from './components/MermaidDiagram.vue'
import './custom.css'

function installVercelObservability(router: Router): void {
  if (!inBrowser) return

  injectAnalytics({ framework: 'vitepress' })
  const speedInsights = injectSpeedInsights({
    framework: 'vitepress',
    route: router.route.path,
  })
  if (speedInsights === null) return

  const previousAfterRouteChange = router.onAfterRouteChange
  router.onAfterRouteChange = async (to) => {
    await previousAfterRouteChange?.(to)
    speedInsights.setRoute(to)
  }
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    installVercelObservability(router)
    app.component('BsgsMatvecPerformance', BsgsMatvecPerformance)
    app.component(
      'BenchmarkV1Compare',
      defineAsyncComponent(
        () => import('./benchmarks/v1/components/BenchmarkCompare.vue'),
      ),
    )
    app.component(
      'BenchmarkV1Portal',
      defineAsyncComponent(
        () => import('./benchmarks/v1/components/BenchmarkPortal.vue'),
      ),
    )
    app.component(
      'BenchmarkV1ResultDetail',
      defineAsyncComponent(
        () => import('./benchmarks/v1/components/BenchmarkResultDetail.vue'),
      ),
    )
    app.component('BlogIndex', BlogIndex)
    app.component('DocCard', DocCard)
    app.component('DocGrid', DocGrid)
    app.component('HomeControlDeck', HomeControlDeck)
    app.component('HomeGraphXray', HomeGraphXray)
    app.component('HomeHero', HomeHero)
    app.component('HomeOpeningRail', HomeOpeningRail)
    app.component('HomeStackBuilder', HomeStackBuilder)
    app.component('InstallCommand', InstallCommand)
    app.component('MermaidDiagram', MermaidDiagram)
  },
} satisfies Theme
