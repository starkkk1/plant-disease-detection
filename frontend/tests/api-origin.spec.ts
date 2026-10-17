import { expect, test } from "@playwright/test";
import { resolveApiBaseUrl } from "../src/lib/prediction-api";

test("HTTPS uses the reverse proxy while HTTP demos retain the API port", () => {
  expect(resolveApiBaseUrl("https://caycachua.id.vn")).toBe("https://caycachua.id.vn/api");
  expect(resolveApiBaseUrl("https://caycachua.id.vn:8443")).toBe("https://caycachua.id.vn:8443/api");
  expect(resolveApiBaseUrl("http://13.239.243.241:3000")).toBe("http://13.239.243.241:8000");
  expect(resolveApiBaseUrl("http://127.0.0.1:3005")).toBe("http://127.0.0.1:8000");
});

test("an explicit API URL takes precedence and removes trailing slashes", () => {
  expect(resolveApiBaseUrl("https://caycachua.id.vn", "https://api.example.com///")).toBe("https://api.example.com");
  expect(resolveApiBaseUrl("http://localhost:3000", "/api/")).toBe("/api");
});
