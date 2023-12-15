import path,{ resolve } from 'path' //todo 
//import path from "path"; //这个path用到了上面安装的@types/node
import { loadEnv } from 'vite'
import { defineConfig, UserConfig, ConfigEnv  } from 'vite';
import vue from '@vitejs/plugin-vue';
import VueSetupExtend from 'vite-plugin-vue-setup-extend';
import AutoImport from 'unplugin-auto-import/vite';
import Components from 'unplugin-vue-components/vite';
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers';

import topLevelAwait from 'vite-plugin-top-level-await' 

export default  defineConfig({
	base: './',
	server: {
		host: '0.0.0.0',
		port: 5174,
		cors:true
	},
	build: {
        target: ['esnext']
    },
	plugins: [
		vue(),
		VueSetupExtend(),
		AutoImport({
			resolvers: [ElementPlusResolver()]
		}),
		Components({
			resolvers: [ElementPlusResolver()]
		}),
		topLevelAwait({
			promiseExportName: '__tla',
			promiseImportName: i => `__tla_${i}`
		}), 
	],
	optimizeDeps: {
		include: ['schart.js']
	},
	 //这里进行配置别名
	 resolve: {
		alias: {
		  "@": path.resolve("./src"), // @代替src
		  "#": path.resolve("./types"), // #代替types
		},
	  }, 
});

const root = process.cwd()
function pathResolve(dir: string) {
	return resolve(root, '.', dir)
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

	}
}