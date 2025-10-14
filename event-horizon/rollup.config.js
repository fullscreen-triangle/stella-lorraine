import babel from '@rollup/plugin-babel';
import resolve from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';

const config = {
  input: 'src/index.js',
  external: ['d3-array', 'd3-axis', 'd3-scale', 'd3-selection', 'd3-shape', 'd3-time', 'd3-time-format', 'd3-transition'],
  output: {
    name: 'd3Horizon',
    globals: {
      'd3-array': 'd3',
      'd3-axis': 'd3',
      'd3-scale': 'd3',
      'd3-selection': 'd3',
      'd3-shape': 'd3',
      'd3-time': 'd3',
      'd3-time-format': 'd3',
      'd3-transition': 'd3'
    }
  },
  plugins: [
    resolve(),
    babel({
      babelHelpers: 'bundled',
      exclude: 'node_modules/**'
    })
  ]
};

export default [
  // UMD build
  {
    ...config,
    output: {
      ...config.output,
      file: 'dist/d3-horizon-chart.js',
      format: 'umd'
    }
  },
  // Minified UMD build
  {
    ...config,
    output: {
      ...config.output,
      file: 'dist/d3-horizon-chart.min.js',
      format: 'umd'
    },
    plugins: [...config.plugins, terser()]
  },
  // ES Module build
  {
    ...config,
    output: {
      file: 'dist/d3-horizon-chart.esm.js',
      format: 'es'
    }
  }
];
