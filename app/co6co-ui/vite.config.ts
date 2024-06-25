import { defineConfig } from 'vite';
import vueJsx from '@vitejs/plugin-vue-jsx';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path'; // 主要用于alias文件路径别名  pnpm install -D path
export default defineConfig({
	plugins: [vue(), vueJsx()],
	/*
	resolve: {
		alias: { '@': resolve(__dirname, './src') }, //npm install --save-dev @types/node
	},
	*/
	build: {
		minify: false, //'esbuild', //压缩方式
		commonjsOptions: {
			defaultIsModuleExports: true,
			sourceMap: true,
		},

		rollupOptions: {
			//// 确保外部化处理那些你不想打包进库的依赖
			external: ['vue', 'element-plus'],
			output: {
				// 在 UMD 构建模式下为这些外部化的依赖提供一个全局变量
				globals: {
					vue: 'Vue',
					'element-plus': 'elementPlus',
				},
			},
		},
		lib: {
			entry: './src/index.ts',
			name: 'co6co',
			fileName: 'co6co',
		},
	},
	esbuild: {
		jsxFactory: 'h',
		jsxFragment: 'Fragment',
	},
});
