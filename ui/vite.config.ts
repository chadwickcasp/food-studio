import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { foodStudioApi } from "./server/api.mjs";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "food-studio-local-api",
      configureServer(server) {
        server.middlewares.use(foodStudioApi);
      },
    },
  ],
  server: {
    port: 4173,
    strictPort: true,
  },
});
