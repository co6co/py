import { defineConfig } from 'vite';
import vueJsx from '@vitejs/plugin-vue-jsx';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path'; // 主要用于alias文件路径别名  pnpm install -D path
export default defineConfig({
	plugins: [vue(), vueJsx()],

	resolve: {
		alias: {
			'@': resolve(__dirname, 'src'),
		}, //npm install --save-dev @types/node
	},
	css: {
		//* css模块化
		modules: {
			// css模块化 文件以.module.[css|less|scss]结尾
			generateScopedName: '[name]__[local]___[hash:base64:5]',
			hashPrefix: 'prefix',
		},
		//* 预编译支持less
		preprocessorOptions: {
			less: {
				// 支持内联 JavaScript
				javascriptEnabled: true,
			},
		},
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

			external: ['vue', 'element-plus'],
			output: {
				// 在 UMD 构建模式下为这些外部化的依赖提供一个全局变量
				globals: {
					vue: 'Vue',
					'element-plus': 'elementPlus',
				},
				/*
				assetFileNames: (assetInfo: PreRenderedAsset) => {
					var info = assetInfo.name.split('.');
					var extType = info[info.length - 1];
					if (
						/\.(mp4|webm|ogg|mp3|wav|flac|aac)(\?.*)?$/i.test(assetInfo.name)
					) {
						extType = 'media';
					} else if (/\.(png|jpe?g|gif|svg)(\?.*)?$/.test(assetInfo.name)) {
						extType = 'img';
					} else if (/\.(woff2?|eot|ttf|otf)(\?.*)?$/i.test(assetInfo.name)) {
						extType = 'fonts';
					}
					return `static/${extType}/[name]-[hash][extname]`;
				},
				chunkFileNames: 'static/js/[name]-[hash].js',
				entryFileNames: 'static/js/[name]-[hash].js',
				*/
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
