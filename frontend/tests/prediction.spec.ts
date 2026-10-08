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

test("empty state, camera input, keyboard actions and no horizontal overflow", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Chưa có ảnh chiếc lá" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Nhận diện chiếc lá" })).toBeDisabled();
  await expect(page.getByLabel("Chụp ảnh lá", { exact: true })).toHaveAttribute("capture", "environment");
  await expect(page.getByRole("button", { name: "Chụp ảnh", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("real end-to-end image prediction with CPU checkpoint", async ({ page }, testInfo) => {
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await expect(page.getByAltText("Ảnh lá đã chọn")).toBeVisible();
  const completed = page.waitForResponse((res) => res.url().endsWith("/predict") && res.request().method() === "POST");
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  const result = await (await completed).json();
  expect(result.is_mock).toBe(false);
  expect(result.model_version).toMatch(/^mobilenetv3_small_100:/);
  expect(result.predictions).toHaveLength(3);
  await expect(page.getByRole("heading", { name: "Đốm vi khuẩn", exact: true })).toBeVisible();
  await expect(page.getByText(result.model_version, { exact: true })).toBeVisible();
  await expect(page.getByText("Kết quả mẫu · Không phải phân tích ảnh này")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  if (testInfo.project.name === "mobile") await page.screenshot({ path: "../reports/sprint-01/frontend-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Nhận diện ảnh khác" }).click();
  await expect(page.getByAltText("Ảnh lá đã chọn")).toHaveCount(0);
});

test("mock mode is explicit and does not call the backend", async ({ page }) => {
  let called = false;
  await page.route("**/predict", (route) => { called = true; return route.abort(); });
  await page.getByLabel("Dùng kết quả mẫu để xem giao diện").check();
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByText("Kết quả mẫu · Không phải phân tích ảnh này")).toBeVisible();
  await expect(page.getByText("demo-mock-v1", { exact: true })).toBeVisible();
  expect(called).toBe(false);
});

test("unsupported files and oversized uploads are rejected before fetch", async ({ page }) => {
  let called = false;
  await page.route("**/predict", (route) => { called = true; return route.abort(); });
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles({ name: "photo.gif", mimeType: "image/gif", buffer: Buffer.from("gif") });
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Chỉ hỗ trợ ảnh JPEG hoặc PNG");
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles({ name: "large.png", mimeType: "image/png", buffer: Buffer.alloc(5 * 1024 * 1024 + 1) });
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Ảnh vượt quá 5 MB");
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
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Không đọc được ảnh này");
  await page.getByRole("button", { name: "Thử lại", exact: true }).click();
  await expect(page.getByText("contract-fixture-v1", { exact: true })).toBeVisible();
  await expect(page.locator(".leaf-confidence")).toContainText("80.0");
});

test("unavailable API shows an error without switching to mock", async ({ page }) => {
  await page.route("**/predict", (route) => route.abort("connectionrefused"));
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Không kết nối được dịch vụ nhận diện");
  await expect(page.getByText("demo-mock-v1", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Dùng kết quả mẫu để xem giao diện")).not.toBeChecked();
});

test("invalid response is handled instead of rendering NaN confidence", async ({ page }) => {
  await page.route("**/predict", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ predictions: [{ probability: .8 }] }) }));
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("kết quả chưa hợp lệ");
});

test("cancel discards the pending response and retains the preview", async ({ page }) => {
  await page.route("**/predict", () => {});
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByRole("heading", { name: "Đang xem ảnh chiếc lá" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Chọn ảnh", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Huỷ nhận diện", exact: true }).click();
  await expect(page.getByAltText("Ảnh lá đã chọn")).toBeVisible();
  await expect(page.getByRole("button", { name: "Nhận diện chiếc lá" })).toBeEnabled();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("request timeout aborts fetch and offers retry", async ({ page }) => {
  await page.clock.install();
  await page.route("**/predict", () => {});
  await page.getByLabel("Chọn ảnh lá", { exact: true }).setInputFiles(fixture);
  await page.getByRole("button", { name: "Nhận diện chiếc lá" }).click();
  await expect(page.getByRole("heading", { name: "Đang xem ảnh chiếc lá" })).toBeVisible();
  await page.clock.fastForward(26_000);
  await expect(page.getByRole("main").getByRole("alert")).toContainText("Nhận diện mất quá nhiều thời gian");
  await expect(page.getByRole("button", { name: "Thử lại", exact: true })).toBeEnabled();
});
