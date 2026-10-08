# Mobile implementation decision — issue #7

Decision: reuse the existing Next.js App Router application as a responsive phone web UI for the eight-week OJT. The MVP runs in a browser; installable PWA features are a later step. No Flutter project existed in the audited repository and none is assumed to exist now.

## Existing code and reuse

The original `frontend/src/app/page.tsx` was a single client component with image selection, object-URL preview, prediction and Qdrant image-search requests, a desktop-oriented two-column layout and inline result cards. It called a fixed loopback API, accepted any image MIME, used `any[]` for results and expected the legacy prediction shape. There was no camera input, request timeout, file limit or distinct API adapter. The layout had local Geist fonts and generic Next.js metadata.

Retained: Next.js routing/build system, React state flow, Tailwind/global styling infrastructure, local fonts, lucide icons, image preview concept and the research search UI. The research page is now `/research`; classification calls use v1 and search uses a configurable separate API. The main page adds an accessible Vietnamese phone layout, gallery/camera inputs, bounded file validation, preview cleanup, explicit states and typed real/mock adapters. Object URLs are revoked and cancelled requests cannot replace a newer image's result.

## Comparison for this repository

| Consideration | A: Next.js responsive, then optional PWA | B: Separate Flutter app |
|---|---|---|
| Existing implementation | Reuses the actual frontend and toolchain | Starts a new project and Dart UI |
| API contract | Same multipart API, typed adapter already implemented | Requires another API client and testing path |
| Phone demo | Browser link on Android/iOS; gallery and capture input | Build/sign an Android app and maintain SDK/emulator setup |
| Delivery work | One UI for phone and desktop | A separate native UI alongside the research frontend |
| Camera/offline/native features | Browser-dependent; offline AI is not implemented | Better access to native APIs, but model/runtime work is additional |
| OJT fit | Fits the current classification MVP and measured backend | Justified later if confirmed requirements require native functionality |

This choice is an engineering judgment based on the code and schedule, not a claim that Flutter is generally unsuitable. Next.js supports a web-app manifest; an installable PWA and service worker would be explicit follow-up work. [Next.js PWA guide](https://nextjs.org/docs/app/guides/progressive-web-apps)

## Delivered phone flow

`/` supports no-image state, JPEG/PNG upload, camera capture input, preview, loading, cancellation, retry, actual results including model version, invalid file/size/HTTP/network/timeout errors and explicitly labeled demo results. A frontend error does not silently fall back to mock data. The camera input uses `capture="environment"`; whether the browser opens a camera or picker depends on the device/browser. [MDN capture attribute](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Attributes/capture)

## Demo target and test limits

Automated target: Chromium through Microsoft Edge on Windows, using Playwright's Pixel 7 viewport/touch/device emulation; desktop viewport 1280×900 is also tested. This is browser device emulation, not a physical Android phone or Android OS emulator. The automated real-image flow uploads `frontend/tests/fixtures/tomato-leaf.jpg` to the actual FastAPI service and CPU checkpoint, verifies the response and rendered label/version, and saves `reports/sprint-01/frontend-mobile.png`.

Physical demo target: Android Chrome on a phone on the development computer's Wi-Fi. Hardware camera behavior and physical-device networking still require that device; they are not falsely recorded as tested. iOS Safari may supply HEIC photos, which the v1 backend rejects; use JPEG/PNG or export the photo first. See the LAN procedure in [RUN_GUIDE.md](../RUN_GUIDE.md).

## Next work within eight weeks

1. Demo on the team's actual Android device and check permission/picker/camera behavior.
2. Confirm field-photo requirements and expand the external evaluation dataset before claiming field accuracy.
3. Once hosting is approved, set HTTPS API origin, allowed CORS origins and configure frontend hosting; rerun dependency audits and review the remaining build/lint findings in [frontend-dependency-review.md](frontend-dependency-review.md).
4. Add manifest/icons/install/offline shell only if PWA installation is required. Offline server inference is unavailable and must not display cached predictions as new results.
5. Revisit Flutter if stakeholders require app-store delivery, reliable native camera controls or on-device inference. These are outside the current web MVP.
