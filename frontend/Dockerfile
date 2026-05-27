# ── Frontend Dockerfile (multi-stage) ───────────────────────────
# Build stage: compile Vite/React assets
# Serve stage: nginx serving static dist
#
# Environment variables (set in docker-compose.yml):
#   API_URL — Backend HTTP URL (default: http://backend:8000)
#   WS_URL  — Backend WebSocket URL (default: ws://backend:8000)

# ================================================================
# BUILD
# ================================================================
FROM node:24-alpine AS build

WORKDIR /app

# Install dependencies (layer caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# ================================================================
# SERVE
# ================================================================
FROM nginx:alpine AS runtime

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx template — entrypoint auto-runs envsubst to /etc/nginx/conf.d/
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
