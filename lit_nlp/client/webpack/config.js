/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const glob = require('glob');
const path = require('path');
const webpack = require('webpack');
const FileManagerPlugin = require('filemanager-webpack-plugin');
const TerserPlugin = require("terser-webpack-plugin");

const GLOB_OPTIONS = {
  ignore: ['**/*test.ts', '**/*test.js', '**/testing_utils.ts']
};

/**
 * WebPack config generator function.
 *
 * @param {?object=} env Environment variables for WebPack to use.
 * @return {!object} WebPack config definition.
 */
module.exports = (env = {}) => {
  console.log('Packing 🔥LIT for the web...', env);
  const isProd = !!env.production;
  const buildDirsStr = env.build || '';

  // File groups to include in the default build entry.
  const core = glob.sync(resolveDir('../core/**/*.ts'), GLOB_OPTIONS);
  const elements = glob.sync(resolveDir('../elements/**/*.ts'), GLOB_OPTIONS);
  const lib = glob.sync(resolveDir('../lib/**/*.ts'), GLOB_OPTIONS);
  const modules = glob.sync(resolveDir('../modules/**/*.ts'), GLOB_OPTIONS);
  const services = glob.sync(resolveDir('../services/**/*.ts'), GLOB_OPTIONS);

   /**
    * The [entry points](https://v4.webpack.js.org/concepts/entry-points/) to
    * build, including the core LIT app bundle, `default`.
    */
  const entry = {
    default: [
      resolveDir('../main.ts'),
      ...core,
      ...elements,
      ...lib,
      ...modules,
      ...services,
    ],
  };

  /**
   * The FileManagerPlugin params, which sepcify how to handle the generted
   * bundles for each path in the `env.build` flag, described below.
   */
  const fileManagerParams = {
    onEnd: {
      copy: [{
        source: resolveDir('../static'),
        destination: resolveDir('../build/default/static')
      }],
      move: [],
    },
  };

  // LIT's build commands (`yarn build`, `yarn watch`) accept an `env.build`
  // flag containing a comma-separated list of directories.
  //
  // Directories are asusmed to be relative paths from the `lit_nlp` directory,
  // hence the `desitnation` resolutions with the `../../` prefix below.
  //
  // A WebPack `entry` is added for each directory in the `env.build` flag,
  // using the paths in the `default` entry as a baseline and adding any paths
  // that match the glob `../../${path}/**/*.ts`. This is how LIT supports
  // adding demo-specific custom modules.
  //
  // WebPack outputs the all bundles into the `lit_nlp/client/build` directory.
  // FileManagerPlugin rules are added to 1) copy LIT's static assets into the
  // build path, 2) move the generated JS bundle into the build path, and 3)
  // delete the WebPack directory for the build path.
  for (const path of buildDirsStr.split(',').filter(p => p.length > 0)) {
    const [moduleName] = path.split('/').slice(-1);

    entry[moduleName] = [
      ...entry.default,
      ...glob.sync(resolveDir(`../../${path}/**/*.ts`), GLOB_OPTIONS)
    ];

    fileManagerParams.onEnd.copy.push({
      source: resolveDir('../static'),
      destination: resolveDir(`../../${path}/build/static`)
    });

    fileManagerParams.onEnd.move.push({
      source: resolveDir(`../build/${moduleName}/main.js`),
      destination: resolveDir(`../../${path}/build/main.js`)
    });

    fileManagerParams.onEnd.delete = fileManagerParams.onEnd.delete || [];
    fileManagerParams.onEnd.delete.push(resolveDir(`../build/${moduleName}`));
  }

  return {
    mode: isProd ? 'production' : 'development',
    devtool: 'source-map',
    module: {
      rules: [
        {
          test: /(\.ts$|\.js$)/,
          exclude: [/node_modules/, '/test.ts$/'],
          use: [
            {
              loader: 'ts-loader',
              options: {
                compilerOptions: {
                  target: 'es6',
                  noImplicitAny: false,
                },
              },
            },
          ],
        },
        // Load the lit-element css files
        {
          test: /\.css$/i,
          loader: resolveDir('./lit-css-loader.js'),
        },
      ],
    },
    optimization: {
      minimize: isProd,
      minimizer: [
        new TerserPlugin({
          cache: true,
          parallel: true,
          sourceMap: true,
          terserOptions: {
            keep_classnames: true   // Required for LIT_TYPES_REGISTRY to work
          }
        })
      ]
    },
    resolve: {
      modules: ['node_modules'],
      extensions: ['.ts', '.js'],
    },
    entry,
    output: {
      filename: '[name]/main.js',
      path: resolveDir('../build'),
    },
    plugins: [
      new webpack.DefinePlugin({
        PRODUCTION: isProd,
      }),
      new FileManagerPlugin(fileManagerParams)
    ],
    watch: !isProd,
  };
};

/**
 * Convenience wrapper for path.resolve().
 *
 * @param {string} relativeDir path to a directory relative to
 *    lit_nlp/client/webpack.
 *
 * @return {string} Fully qualified path.
 */
function resolveDir(relativeDir) {
  return path.resolve(__dirname, relativeDir);
}
