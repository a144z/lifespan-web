/** @type {import('next').NextConfig} */
const webpack = require('webpack');
const path = require('path');

const nextConfig = {
  reactStrictMode: true,
  // Exclude Node.js ONNX files from static file tracing
  experimental: {
    outputFileTracingExcludes: {
      '*': [
        '**/node_modules/onnxruntime-web/dist/ort.node.min.mjs',
        '**/node_modules/onnxruntime-web/**/ort.node.min.mjs',
      ],
    },
  },
  // Ensure large static files are served correctly
  staticPageGenerationTimeout: 120,
  // Optimize for ONNX.js
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
      
      // Ignore Node.js-specific files in onnxruntime-web
      config.resolve.alias = {
        ...config.resolve.alias,
        'onnxruntime-node': false,
        // Replace Node.js ONNX file with empty module at resolve stage
        './ort.node.min.mjs': require.resolve('./lib/empty-module.js'),
        'ort.node.min.mjs': require.resolve('./lib/empty-module.js'),
      };
      // Hard block the node build via externals (prevents bundling/minifying)
      config.externals = config.externals || [];
      config.externals.push(({ request }, callback) => {
        if (request && request.includes('ort.node.min.mjs')) return callback(null, '{}');
        return callback();
      });
      
      // Use IgnorePlugin to completely ignore Node.js ONNX files
      config.plugins.push(
        new webpack.IgnorePlugin({
          checkResource(resource, context) {
            // Ignore any file containing ort.node.min.mjs
            if (/ort\.node\.min\.mjs/.test(resource)) {
              return true;
            }
            return false;
          },
        })
      );
      
      // Use NormalModuleReplacementPlugin as fallback
      config.plugins.push(
        new webpack.NormalModuleReplacementPlugin(
          /ort\.node\.min\.mjs$/,
          require.resolve('./lib/empty-module.js')
        )
      );
      
      // Exclude from module rules processing - must be first
      // Treat as ignored asset and do not emit
      config.module.rules.unshift({
        test: /ort\.node\.min\.mjs(\?.*)?$/,
        type: 'asset/source',
        parser: { javascript: { importMeta: false } },
        generator: { emit: false },
      });
      
      // Configure webpack to not parse these files
      if (!config.module.noParse) {
        config.module.noParse = [];
      }
      config.module.noParse.push(/ort\.node\.min\.mjs$/);
      
      // Filter out Node.js files from assets at the earliest stage
      config.plugins.push({
        apply: (compiler) => {
          compiler.hooks.compilation.tap('RemoveNodeFiles', (compilation) => {
            // Remove at the earliest stage before any processing
            compilation.hooks.processAssets.tap(
              {
                name: 'RemoveNodeFiles',
                stage: webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONAL,
              },
              (assets) => {
                Object.keys(assets).forEach((filename) => {
                  if (/ort\.node\.min\.mjs/.test(filename)) {
                    delete assets[filename];
                  }
                });
              }
            );
          });
        },
      });
      
      // Configure Terser to exclude Node.js files using test function
      if (config.optimization && config.optimization.minimizer) {
        config.optimization.minimizer.forEach((minimizer) => {
          if (minimizer.constructor.name === 'TerserPlugin') {
            const originalTest = minimizer.options.test;
            minimizer.options = {
              ...minimizer.options,
              test: (filename) => {
                // Exclude Node.js ONNX files
                if (/ort\.node\.min\.mjs/.test(filename)) {
                  return false;
                }
                // Use original test if it exists
                if (originalTest) {
                  if (typeof originalTest === 'function') {
                    return originalTest(filename);
                  }
                  if (originalTest instanceof RegExp) {
                    return originalTest.test(filename);
                  }
                }
                return true;
              },
            };
          }
        });
      }
    }
    return config;
  },
  // Enable static file serving for ONNX models and WASM files
  async headers() {
    return [
      {
        source: '/models/:path*',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/octet-stream',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
        ],
      },
      {
        source: '/:path*.onnx',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/octet-stream',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        source: '/wasm/:path*',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        source: '/:path*.wasm',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;

