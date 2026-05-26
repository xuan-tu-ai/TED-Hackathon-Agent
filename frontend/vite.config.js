import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'https://app-ted-hackathon-be-dev-001.azurewebsites.net',
          changeOrigin: true,
        },
        '/ws': {
          target: env.VITE_WS_URL || 'wss://app-ted-hackathon-be-dev-001.azurewebsites.net',
          ws: true,
        },
      },
    },
  };
});
