import { defineConfig, devices } from "@playwright/test";

const python = process.env.PYTHON_EXECUTABLE || "python";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 40_000,
  reporter: [["list"], ["html", { open: "never" }]],
  use: { baseURL: "http://127.0.0.1:3005", trace: "retain-on-failure",
         screenshot: "only-on-failure", channel: process.platform === "win32" ? "msedge" : "chromium" },
  projects: [
    { name: "mobile", use: { ...devices["Pixel 7"], defaultBrowserType: "chromium" } },
    { name: "desktop", use: { viewport: { width: 1280, height: 900 } } },
  ],
  webServer: [
    { command: `"${python}" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`, cwd: "..",
      url: "http://127.0.0.1:8000/health", reuseExistingServer: !process.env.CI, timeout: 120_000,
      env: { CORS_ORIGINS: "http://127.0.0.1:3005" } },
    { command: "npm run start -- --hostname 127.0.0.1 --port 3005", url: "http://127.0.0.1:3005",
      reuseExistingServer: !process.env.CI, timeout: 120_000 },
  ],
});
