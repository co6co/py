import path, { resolve } from 'path';
import { loadEnv } from 'vite';
import { defineConfig, UserConfig, ConfigEnv } from 'vite';
import vue from '@vitejs/plugin-vue';
import VueSetupExtend from 'vite-plugin-vue-setup-extend'; 
import vueJs from '@vitejs/plugin-vue-jsx'
import AutoImport from 'unplugin-auto-import/vite';
import Components from 'unplugin-vue-components/vite';
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'; 
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
	base: './UI',
	//server:{hmr:{overlay:false} },
	//publicDir:"/public",//静态文件目录
	server: {
		host: '0.0.0.0',
		port: 5173,
	},
	//esbuild:{jsx: "preserve"},
	plugins: [
		vue(),
		VueSetupExtend(), 
		vueJs(),
		AutoImport({
			resolvers: [ElementPlusResolver()],
		}),
		Components({
			resolvers: [ElementPlusResolver()],
		}),
		topLevelAwait({
			promiseExportName: '__tla',
			promiseImportName: (i) => `__tla_${i}`,
		}),
		//"@vue/babel-plugin-jsx"
	],
	optimizeDeps: {
		include: ['schart.js'],
	}, //这里进行配置别名
	resolve: {
		//extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json', '.less', '.css'],
		alias: {
			'@': path.resolve('./src'), // @代替src
			'#': path.resolve('./types'), // #代替types
		},
	},
	build: {
		rollupOptions: {
			input: {
				pbIndex: path.resolve(__dirname, 'index.html'),
				mpIndex: path.resolve(__dirname, 'home.html'),
			},
		},
	},
});

const root = process.cwd();
function pathResolve(dir: string) {
	return resolve(root, '.', dir);
}

export const useConfig = ({ command, mode }: ConfigEnv): UserConfig => {
	let env = {} as any;
	const isBuild = command === 'build';
	console.warn("编译参数:",process.argv)
	if (isBuild) {
		env = loadEnv(mode, root);		
	} else { 
		env = loadEnv(
			process.argv[3] === '--mode' ? process.argv[4] : process.argv[3],
			root
		);
	}
	return {
		base: env.API_URL_BASE,
	};
};
