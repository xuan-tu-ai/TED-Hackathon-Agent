# ── Frontend Dockerfile (multi-stage) ───────────────────────────
# Build stage: compile Vite/React assets
# Serve stage: nginx serving static dist

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

# Copy nginx config with API/WS proxy to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
