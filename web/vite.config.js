import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/static/",
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/assets": "http://127.0.0.1:8765",
      "/docs": "http://127.0.0.1:8765",
      "/openapi.json": "http://127.0.0.1:8765",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
