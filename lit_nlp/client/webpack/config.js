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
const path = require('path');
const webpack = require('webpack');
const litCssLoader = require('./lit-css-loader');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
module.exports = (env = {}) => {
  const isProd = !!env.production;
  const isDev = !isProd;
  console.log('⭐️ Packing web...');
  console.log('🍕', { isProd });
  const cssExtractPlugin = new MiniCssExtractPlugin({
    filename: '[name].css',
  });
  return {
    mode: isDev ? 'development' : 'production',
    devtool: isDev ? 'inline-source-map' : 'none',
    module: {
      rules: [
        {
          test: /(\.ts$|\.js$)/,
          exclude: [/node_modules/, '/test.ts$/', '/umap_worker/'],
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
    resolve: {
      modules: ['node_modules'],
      extensions: ['.ts', '.js'],
    },
    entry: resolveDir('../main.ts'),
    output: {
      filename: '[name].js',
      path: resolveDir('../build'),
    },
    plugins: [
      new webpack.DefinePlugin({
        PRODUCTION: isProd,
      }),
      cssExtractPlugin,
      new HtmlWebpackPlugin({
        template: resolveDir(`../static/index.html`),
        filename: resolveDir(`../build/index.html`),
      }),
      new CopyWebpackPlugin([{from: 'static', to: 'static'}]),
    ],
    watch: isDev,
  };
};
function insertIf(condition, ...elements) {
  return condition ? elements : [];
}
function resolveDir(relativeDir) {
  return path.resolve(__dirname, relativeDir);
}

