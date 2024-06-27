import { defineConfig } from 'vite';
import { PreRenderedAsset } from 'rollup';
import vueJsx from '@vitejs/plugin-vue-jsx';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path'; // 主要用于alias文件路径别名  pnpm install -D path
import packageJson from './package.json';
import dts from 'vite-plugin-dts'; //打包为 .d.ts 文件
export default defineConfig({
	plugins: [
		vue(),
		vueJsx(),
		dts({
			include: ['src/**/*'],
			copyDtsFiles: true,
		}),
	],

	resolve: {
		alias: {
			'@': resolve(__dirname, 'src'),
		}, //npm install --save-dev @types/node
	},

	build: {
		minify: false, //'esbuild', //压缩方式
		commonjsOptions: {
			defaultIsModuleExports: true,
			sourceMap: true,
		},
		copyPublicDir: true,
		rollupOptions: {
			//// 确保外部化处理那些你不想打包进库的依赖
			//external: ['vue', 'element-plus'],
			output: {
				// 在 UMD 构建模式下为这些外部化的依赖提供一个全局变量
				globals: {
					vue: 'Vue',
					'element-plus': 'elementPlus',
				},
				//静态资源输出配置
				assetFileNames(assetInfo: PreRenderedAsset) {
					if (assetInfo.name) {
						//css文件单独
						if (assetInfo.name.endsWith('.css')) {
							return `index.css`;
						}
						//图片文件
						else if (
							['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'].some((ext) =>
								assetInfo.name!.endsWith(ext)
							)
						) {
							return `static/img/[name]-[hash].[ext]`;
						}
						//其他资源输出
						else {
							return `static/[name]-[hash].[ext]`;
						}
					} else {
						return `static/[name]-[hash].[ext]`;
					}
				},
			},
		},
		lib: {
			entry: './src/index.ts',
			name: packageJson.name,
			formats: ['es', 'cjs', 'umd'],
			fileName: (format, e) => `index-${format}.js`,
		},
	},
	/*
	esbuild: {
		jsxFactory: 'h',
		jsxFragment: 'Fragment',
	},*/
});
