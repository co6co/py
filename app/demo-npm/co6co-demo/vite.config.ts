import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import vueJsx from '@vitejs/plugin-vue-jsx';
import path from 'path';

import autoImport from 'unplugin-auto-import/vite';
import components from 'unplugin-vue-components/vite'; //按需引入
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'; //按需引入 element-plus
import topLevelAwait from 'vite-plugin-top-level-await'; //解决 错误 Top-level await is not available

const resolve = (dir: string) => path.join(__dirname, dir);

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [
		vue(),
		vueJsx(),
		components({
			resolvers: [ElementPlusResolver()],
		}),
		autoImport({
			resolvers: [ElementPlusResolver()],
		}),
		topLevelAwait({
			promiseExportName: '__tla',
			promiseImportName: (i) => `__tla_${i}`,
		}),
	],
	resolve: {
		alias: {
			'@':path.resolve('./src'), // 相对路径别名配置，使用@代替src
			'@e': resolve('./examples'),
			'@com':path.resolve('./src/components'),
		},
	},

	build: {
		rollupOptions: {
			// 请确保外部化那些你的库中不需要的依赖
			external: ['vue'],
			output: {
				// 在 UMD 构建模式下为这些外部化的依赖提供一个全局变量
				globals: {
					vue: 'Vue',
				},
			},
		},
		lib: {
			entry: 'src/index.ts',
			name: 'co6co-demo',
			fileName: (format) => `co6co-demo.${format}.js`,
		},
	},
});
