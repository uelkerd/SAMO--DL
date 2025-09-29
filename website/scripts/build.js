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

console.log('🔨 Building SAMO-DL website...');

// Source and destination paths
const configPath = path.join(__dirname, '..', 'js', 'config.js');

// Read the config file
let configContent = fs.readFileSync(configPath, 'utf8');

// Inject build-time variables
if (isProduction) {
  console.log('📦 Production build - enabling authentication');
  // Replace the REQUIRE_AUTH line to inject window.PROD_REQUIRE_AUTH = true
  configContent = configContent.replace(
    'REQUIRE_AUTH: (typeof window.PROD_REQUIRE_AUTH !== \'undefined\') ? window.PROD_REQUIRE_AUTH : true // Build-time injected for production',
    'REQUIRE_AUTH: true // Production build - authentication required'
  );

  // Add the global injection at the top of the config
  configContent = '// Build-time injected production configuration\nwindow.PROD_REQUIRE_AUTH = true;\n\n' + configContent;

} else if (isDevelopment) {
  console.log('🛠️  Development build - disabling authentication for convenience');
  // For development, leave REQUIRE_AUTH as false via localhost override
  configContent = configContent.replace(
    'REQUIRE_AUTH: (typeof window.PROD_REQUIRE_AUTH !== \'undefined\') ? window.PROD_REQUIRE_AUTH : true // Build-time injected for production',
    'REQUIRE_AUTH: false // Development build - authentication disabled'
  );
} else {
  console.log('⚠️  No build type specified, using default configuration');
}

// Write back to the same file (for static site deployment)
fs.writeFileSync(configPath, configContent);

console.log('✅ Build complete!');
console.log(`   REQUIRE_AUTH: ${isProduction ? 'true (production)' : isDevelopment ? 'false (development)' : 'default (secure)'}`);
console.log('   Config file updated successfully');
