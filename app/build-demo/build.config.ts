import { defineBuildConfig } from 'unbuild';

export default defineBuildConfig({
  entries: ['src/index.ts'],
  outDir: 'dist',
  clean: true,
  sourcemap: true,
  declaration: true,
  //stub: true,
  //name: 'co6co',
});
