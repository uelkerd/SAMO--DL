# SAMO-DL Website Build Scripts

This directory contains build scripts for the SAMO-DL website frontend.

## Build Process

The website uses build-time configuration injection to securely handle authentication
settings.

### Build Commands

```bash
# Production build (enables authentication)
npm run build:prod

# Development build (disables authentication for convenience)
npm run build:dev

# Default build (secure defaults)
npm run build
```

### What the Build Script Does

The build script (`scripts/build.js`) modifies `js/config.js` to inject authentication
settings:

- **Production builds**: Sets `REQUIRE_AUTH: true` and injects
  `window.PROD_REQUIRE_AUTH = true`
- **Development builds**: Sets `REQUIRE_AUTH: false` for local development convenience
- **Default builds**: Uses secure defaults with build-time injection capability

### Security Model

1. **Client-side**: `REQUIRE_AUTH` controls whether the frontend requires authentication
2. **Server-side**: All protected endpoints use `@require_api_key` decorator for
   server-side validation
3. **Build-time**: Production builds inject `window.PROD_REQUIRE_AUTH = true` to enforce
   authentication

### CI/CD Integration

For production deployments, use:

```yaml
# Example GitHub Actions
- name: Build for production
  run: npm run build:prod
  env:
    NODE_ENV: production

- name: Deploy to production
  # ... deployment steps
```

### Local Development

For local development, use:

```bash
npm run build:dev  # or just run without building for localhost override
```

The localhost environment detection will automatically disable authentication for
convenience.
