import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'https://app-ted-hackathon-be-dev-001.azurewebsites.net',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://app-ted-hackathon-be-dev-001.azurewebsites.net',
        ws: true,
      },
    },
  },
});
