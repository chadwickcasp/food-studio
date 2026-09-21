/// <reference types="vite/client" />

declare module "../server/api.mjs" {
  import type { Connect } from "vite";
  export const foodStudioApi: Connect.NextHandleFunction;
}
