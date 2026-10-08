# Frontend dependency review

Validated on 2026-10-08 after implementing the local phone UI and API integration.

- Upgraded Next.js and `eslint-config-next` together from 14.2.15 to **15.5.27**. React 18 is retained; the production build and real-model browser tests verify this application on the new version.
- Applied compatible dependency patches and regenerated the lockfile. A clean `npm ci --ignore-scripts` completed successfully.
- Set PostCSS to `^8.5.29` and override Next.js's pinned PostCSS with the same dependency using `$postcss`. This keeps the same PostCSS major and avoids retaining the vulnerable nested 8.4.31 copy. `npm ls postcss --depth 1` confirms 8.5.29 is shared by Next.js and Tailwind.
- Set TypeScript's target to ES2017 as required by the upgraded Next.js build.

Next.js 15 is a maintenance release under the [official support policy](https://nextjs.org/support-policy). The [official upgrade guide](https://nextjs.org/docs/app/guides/upgrading/version-15) describes the major-version changes. This application has no cookies/headers/dynamic-route server code needing the asynchronous request-API migration.

## Audit evidence

| Check | Result |
|---|---|
| Before changes, all dependencies | 16 findings: 1 critical, 13 high, 2 moderate |
| After changes, runtime (`npm audit --omit=dev`) | **0 reported vulnerabilities** |
| After changes, all dependencies | 9 findings: 0 critical, 7 high, 2 moderate |

The final raw reports are [frontend-runtime-audit.json](../reports/sprint-01/frontend-runtime-audit.json) and [frontend-dependency-audit.json](../reports/sprint-01/frontend-dependency-audit.json). These are npm's advisory results at the time of validation, not a guarantee that all future vulnerabilities are absent.

Remaining findings involve build/lint tools: `braces` through glob matching in Tailwind/ESLint, and `postcss-selector-parser` through Tailwind's build plugins. The report counts affected parent packages as well as the underlying packages. Do not process untrusted glob patterns or styles through these tools. Their remediation is a separate build-tool migration; `npm audit fix --force` would change major versions and does not constitute a verified fix for this project. No findings were suppressed or removed from the reports.

## Reproduce

From `frontend/`, run `npm ci`, `npm run build`, `npm audit --omit=dev`, and `npm audit`. Set `PYTHON_EXECUTABLE` to the ML environment and run `npm run test:e2e` with the real local checkpoint present. The automated browser run starts its own local API and frontend; it uses no AWS services.
