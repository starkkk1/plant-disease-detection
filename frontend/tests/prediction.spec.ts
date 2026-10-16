import { expect, test } from "@playwright/test";
import path from "node:path";

const fixture = path.join(__dirname, "fixtures/tomato-leaf.jpg");
const response = {
  request_id: "bb7bc2ad-3eb5-4f21-89b7-613b29d5c907", model_version: "contract-fixture-v1", is_mock: false,
  processing_time_ms: 15,
  predictions: [
    { class_id: 0, label: "Tomato___Bacterial_spot", confidence: .8 },
    { class_id: 1, label: "Tomato___Early_blight", confidence: .15 },
    { class_id: 9, label: "Tomato___healthy", confidence: .05 },
  ],
};

test.beforeEach(async ({ page }) => { await page.goto("/"); });

test("empty state, camera input, keyboard actions and no horizontal overflow", async ({ page }, testInfo) => {
  await expect(page.getByRole("heading", { name: "Your results, here" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyze leaf" })).toBeDisabled();
  await expect(page.getByLabel("Take leaf photo", { exact: true })).toHaveAttribute("capture", "environment");
  await expect(page.getByRole("button", { name: "Take photo", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.getByLabel("Use sample results")).toBeHidden();
  await page.screenshot({ path: testInfo.outputPath("empty.png"), fullPage: true, animations: "disabled" });
});

test("real end-to-end prediction and Grad-CAM++ with CPU checkpoint", async ({ page }, testInfo) => {
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await expect(page.getByAltText("Selected leaf")).toBeVisible();
  const completed = page.waitForResponse((res) => res.url().endsWith("/predict") && res.request().method() === "POST");
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  const result = await (await completed).json();
  expect(result.is_mock).toBe(false);
  expect(result.model_version).toMatch(/^mobilenetv3_small_100:/);
  expect(result.predictions).toHaveLength(3);
  await expect(page.getByRole("heading", { name: "Bacterial spot", exact: true })).toBeVisible();
  await page.getByText("Analysis details", { exact: true }).click();
  await expect(page.getByText(result.model_version, { exact: true })).toBeVisible();
  await expect(page.getByText("Sample results · Not an analysis of this image")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath("prediction.png"), fullPage: true, animations: "disabled" });
  const explained = page.waitForResponse((res) => res.url().endsWith("/explain") && res.request().method() === "POST");
  await page.getByRole("button", { name: "Explain prediction", exact: true }).click();
  const explainResponse = await explained;
  expect(explainResponse.status()).toBe(200);
  const explanation = await explainResponse.json();
  expect(explanation.model_version).toBe(result.model_version);
  expect(explanation.explanation.target.class_id).toBe(result.predictions[0].class_id);
  expect(explanation.explanation.method).toBe("gradcam++");
  const overlay = page.getByAltText("Grad-CAM++ overlay on the leaf");
  await expect(overlay).toBeVisible();
  await expect.poll(() => overlay.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBe(224);
  await expect(page.getByRole("heading", { name: "Bacterial spot", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath("explanation.png"), fullPage: true, animations: "disabled" });
  await page.getByRole("button", { name: "Analyze another leaf" }).click();
  await expect(page.getByAltText("Selected leaf")).toHaveCount(0);
});

test("mock mode is explicit and does not call the backend", async ({ page }) => {
  let called = false;
  await page.route("**/predict", (route) => { called = true; return route.abort(); });
  await page.getByText("Preview options", { exact: true }).click();
  await page.getByLabel("Use sample results").check();
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByText("Sample results · Not an analysis of this image")).toBeVisible();
  await page.getByText("Analysis details", { exact: true }).click();
  await expect(page.getByText("demo-mock-v1", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Explain prediction", exact: true })).toHaveCount(0);
  expect(called).toBe(false);
});

test("explanation errors preserve prediction and support retry", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(response) }));
  let attempts = 0;
  await page.route("**/explain", (route) => {
    attempts++;
    expect(route.request().postData()).toContain('name="class_id"');
    return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({
      request_id: response.request_id, error: { code: "explanation_unavailable", message: "Unavailable" },
    }) });
  });
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await page.getByRole("button", { name: "Explain prediction", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Grad-CAM++ is unavailable");
  await expect(page.getByRole("heading", { name: "Bacterial spot", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retry explanation", exact: true }).click();
  await expect.poll(() => attempts).toBe(2);
});

test("malformed explanation is rejected while keeping the result", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(response) }));
  await page.route("**/explain", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...response, explanation: {} }) }));
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await page.getByRole("button", { name: "Explain prediction", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("invalid result");
  await expect(page.getByAltText("Grad-CAM++ overlay on the leaf")).toHaveCount(0);
  await page.getByText("Analysis details", { exact: true }).click();
  await expect(page.getByText(response.model_version, { exact: true })).toBeVisible();
});

test("cancel explanation keeps prediction and clears loading", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(response) }));
  await page.route("**/explain", () => {});
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await page.getByRole("button", { name: "Explain prediction", exact: true }).click();
  await expect(page.getByRole("button", { name: "Choose image", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Cancel explanation", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Bacterial spot", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Explain prediction", exact: true })).toBeEnabled();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("unsupported files and oversized uploads are rejected before fetch", async ({ page }) => {
  let called = false;
  await page.route("**/predict", (route) => { called = true; return route.abort(); });
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles({ name: "photo.gif", mimeType: "image/gif", buffer: Buffer.from("gif") });
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Only JPEG or PNG images are supported");
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles({ name: "large.png", mimeType: "image/png", buffer: Buffer.alloc(5 * 1024 * 1024 + 1) });
  await expect(page.getByRole("main").getByRole("alert")).toContainText("This image exceeds 5 MB");
  expect(called).toBe(false);
});

test("HTTP image error supports retry and reads the new result schema", async ({ page }) => {
  let attempts = 0;
  await page.route("**/predict", async (route) => {
    attempts++;
    expect(route.request().headers()["content-type"]).toContain("multipart/form-data; boundary=");
    expect(route.request().postData()).toContain('name="file"');
    await route.fulfill({ status: attempts === 1 ? 400 : 200, contentType: "application/json",
      body: JSON.stringify(attempts === 1 ? { request_id: response.request_id, error: { code: "invalid_image", message: "Invalid image" } } : response) });
  });
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("This image could not be read");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await page.getByText("Analysis details", { exact: true }).click();
  await expect(page.getByText("contract-fixture-v1", { exact: true })).toBeVisible();
  await expect(page.locator(".leaf-confidence")).toContainText("80.0");
});

test("unavailable API shows an error without switching to mock", async ({ page }) => {
  await page.route("**/predict", (route) => route.abort("connectionrefused"));
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Cannot connect to the analysis service");
  await expect(page.getByText("demo-mock-v1", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Use sample results")).not.toBeChecked();
});

test("invalid response is handled instead of rendering NaN confidence", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ predictions: [{ probability: .8 }] }) }));
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("invalid result");
});

test("legacy search API is identified rather than accepted as a real prediction", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json",
    body: JSON.stringify({ model_status: "loaded", predictions: [
      { class_name: "Tomato___Bacterial_spot", probability: .84, gradcam_path: "legacy-file.jpg" },
    ] }) }));
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("outdated API version");
  await expect(page.getByText("demo-mock-v1", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Explain prediction", exact: true })).toHaveCount(0);
});

test("cancel discards the pending response and retains the preview", async ({ page }) => {
  await page.route("**/predict", () => {});
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("heading", { name: "Looking at your leaf" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose image", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Cancel analysis", exact: true }).click();
  await expect(page.getByAltText("Selected leaf")).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyze leaf" })).toBeEnabled();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("request timeout aborts fetch and offers retry", async ({ page }) => {
  await page.clock.install();
  await page.route("**/predict", () => {});
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf" }).click();
  await expect(page.getByRole("heading", { name: "Looking at your leaf" })).toBeVisible();
  await page.clock.fastForward(26_000);
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Analysis took too long");
  await expect(page.getByRole("button", { name: "Try again", exact: true })).toBeEnabled();
});

test("keyboard upload, balanced cards and reduced motion", async ({ page }, testInfo) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(response) }));
  const chooser = page.waitForEvent("filechooser");
  const chooseButton = page.getByRole("button", { name: "Choose image", exact: true });
  await chooseButton.focus();
  await chooseButton.press("Enter");
  await (await chooser).setFiles(fixture);
  await expect(page.getByAltText("Selected leaf")).toBeVisible();
  const cards = await page.locator(".leaf-card").evaluateAll((elements) => elements.map((element) => {
    const { x, y, width, height } = element.getBoundingClientRect();
    return { x, y, width, height };
  }));
  if (testInfo.project.name === "desktop") {
    expect(cards[0].y).toBe(cards[1].y);
    expect(cards[0].width).toBe(cards[1].width);
    expect(cards[0].height).toBe(cards[1].height);
  } else {
    expect(cards[1].y).toBeGreaterThan(cards[0].y + cards[0].height);
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  const analyzeButton = page.getByRole("button", { name: "Analyze leaf", exact: true });
  await expect(analyzeButton).toHaveCSS("transition-duration", "0s");
  await analyzeButton.click();
  await expect(page.getByRole("heading", { name: "Bacterial spot", exact: true })).toBeVisible();
  await expect(page.locator(".leaf-result-data")).toHaveCSS("animation-name", "none");
});

test("narrow screen and long disease names stay within the workspace", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
    ...response, predictions: [{ class_id: 7, label: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", confidence: .98 }],
  }) }));
  await page.getByLabel("Choose leaf image", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Analyze leaf", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Tomato yellow leaf curl virus", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});
