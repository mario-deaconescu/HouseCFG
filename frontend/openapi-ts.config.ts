import { defineConfig } from "@hey-api/openapi-ts";

import apiBase from "./src/config/api.ts";

export default defineConfig({
  input: `${apiBase}/openapi.json`,
  output: "src/api",
  plugins: ["@hey-api/client-fetch"],
});
