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

// Define simple patterns for known REQUIRE_AUTH formats
const requireAuthPatterns = [
  /^REQUIRE_AUTH:\s*\(typeof window\.PROD_REQUIRE_AUTH !== ['"]undefined['"]\)\s*\?\s*window\.PROD_REQUIRE_AUTH\s*:\s*true\s*(?:\/\/.*)?$/,
  /^REQUIRE_AUTH:\s*true\s*(?:\/\/.*)?$/,
  /^REQUIRE_AUTH:\s*false\s*(?:\/\/.*)?$/
];

// Split config into lines and find the REQUIRE_AUTH line
const configLines = configContent.split('\n');
let found = false;

for (let i = 0; i < configLines.length; i++) {
  for (const pattern of requireAuthPatterns) {
    if (pattern.test(configLines[i])) {
      found = true;
      if (isProduction) {
        console.log('📦 Production build - enabling authentication');
        configLines[i] = 'REQUIRE_AUTH: true // Production build - authentication required';
        // Prepend production header after replacement
        configContent = PRODUCTION_HEADER + configLines.join('\n');
      } else if (isDevelopment) {
        console.log('🛠️  Development build - disabling authentication for convenience');
        configLines[i] = 'REQUIRE_AUTH: false // Development build - authentication disabled';
        configContent = configLines.join('\n');
      } else {
        console.log('⚠️  No build type specified, using default configuration');
        configLines[i] = "REQUIRE_AUTH: (typeof window.PROD_REQUIRE_AUTH !== 'undefined') ? window.PROD_REQUIRE_AUTH : true // Build-time injected for production";
        configContent = configLines.join('\n');
      }
      break;
    }
  }
  if (found) {
    break;
  }
}

if (!found) {
  throw new Error('Unable to locate REQUIRE_AUTH in config.js');
}

// Write back to the same file (for static site deployment)
fs.writeFileSync(configPath, configContent);

console.log('✅ Build complete!');
console.log(`   REQUIRE_AUTH: ${isProduction ? 'true (production)' : isDevelopment ? 'false (development)' : 'default (secure)'}`);
console.log('   Config file updated successfully');
