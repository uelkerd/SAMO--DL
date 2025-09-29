#!/usr/bin/env node
/**
 * Build script for SAMO-DL website
 * Injects build-time configuration variables
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Parse command line arguments
const args = process.argv.slice(2);
const isProduction = args.includes('--prod');
const isDevelopment = args.includes('--dev');

// Validate mutual exclusivity of build flags
if (isProduction && isDevelopment) {
  console.error('❌ Error: Cannot specify both --prod and --dev flags');
  console.error('   Use either --prod for production build or --dev for development build');
  process.exit(1);
}

console.log('🔨 Building SAMO-DL website...');

// Source and destination paths
const configPath = path.join(__dirname, '..', 'js', 'config.js');

// Read the config file
let configContent = fs.readFileSync(configPath, 'utf8');

// Normalize and inject build-time variables
const PRODUCTION_HEADER = '// Build-time injected production configuration\nwindow.PROD_REQUIRE_AUTH = true;\n\n';

// Strip any previously injected production header
configContent = configContent.replace(
  /^\/\/ Build-time injected production configuration\nwindow\.PROD_REQUIRE_AUTH = true;\n\n/,
  ''
);

// Match the REQUIRE_AUTH line in any of its known forms
const requireAuthPattern =
  /REQUIRE_AUTH:\s*(?:\(typeof window\.PROD_REQUIRE_AUTH !== 'undefined'\)\s*\?\s*window\.PROD_REQUIRE_AUTH\s*:\s*true|true|false)\s*\/\/.*$/m;

if (!requireAuthPattern.test(configContent)) {
  throw new Error('Unable to locate REQUIRE_AUTH in config.js');
}

if (isProduction) {
  console.log('📦 Production build - enabling authentication');
  configContent =
    PRODUCTION_HEADER +
    configContent.replace(
      requireAuthPattern,
      'REQUIRE_AUTH: true // Production build - authentication required'
    );
} else if (isDevelopment) {
  console.log('🛠️  Development build - disabling authentication for convenience');
  configContent = configContent.replace(
    requireAuthPattern,
    'REQUIRE_AUTH: false // Development build - authentication disabled'
  );
} else {
  console.log('⚠️  No build type specified, using default configuration');
  configContent = configContent.replace(
    requireAuthPattern,
    'REQUIRE_AUTH: (typeof window.PROD_REQUIRE_AUTH !== \'undefined\') ? window.PROD_REQUIRE_AUTH : true // Build-time injected for production'
  );
}

// Write back to the same file (for static site deployment)
fs.writeFileSync(configPath, configContent);

console.log('✅ Build complete!');
console.log(`   REQUIRE_AUTH: ${isProduction ? 'true (production)' : isDevelopment ? 'false (development)' : 'default (secure)'}`);
console.log('   Config file updated successfully');
