import { fileURLToPath, URL } from 'node:url'
import path, { resolve } from 'path';
import { loadEnv } from 'vite';
import { defineConfig, UserConfig, ConfigEnv } from 'vite';
import vue from '@vitejs/plugin-vue';  
import vueJsx from '@vitejs/plugin-vue-jsx'
import AutoImport from 'unplugin-auto-import/vite';
import Components from 'unplugin-vue-components/vite';
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'; 
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
	base: './',
	//server:{hmr:{overlay:false} },
	//publicDir:"/public",//静态文件目录
	server: {
		host: '0.0.0.0',
		port: 5173,
	},
	//esbuild:{jsx: "preserve"},
	plugins: [
		vue(),
		Components({
		resolvers: [ElementPlusResolver()]
	}),
	AutoImport({
		resolvers: [ElementPlusResolver()]
	}),
		topLevelAwait({
			promiseExportName: '__tla',
			promiseImportName: (i) => `__tla_${i}`,
		}),
		//"@vue/babel-plugin-jsx"
	],
 

	resolve: {
	 alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
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

export const useConfig= ({ command, mode }: ConfigEnv): UserConfig => {
	let env = {} as any
	const isBuild = command === 'build'
	if (!isBuild) {
	  env = loadEnv((process.argv[3] === '--mode' ? process.argv[4] : process.argv[3]), root)
	} else {
	  env = loadEnv(mode, root)
	}
	return {
		base: env.API_URL_BASE,
	};
};
