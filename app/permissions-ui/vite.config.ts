import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'path'
import { loadEnv } from 'vite'
import { defineConfig, UserConfig, ConfigEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite' //按需引入
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers' //按需引入 element-plus
import topLevelAwait from 'vite-plugin-top-level-await' //解决 错误 Top-level await is not available

// https://vitejs.dev/config/

const root = process.cwd()
function pathResolve(dir: string) {
  return resolve(root, '.', dir)
}
export default (option:{model:string}) => {
  process.env = { ...process.env, ...loadEnv(option.mode, process.cwd()) }

  return defineConfig({
    base: process.env.VITE_APP_BASE, //生成 引用 css 文件和js 文件 增加的前缀
    build: {
      outDir: process.env.VITE_APP_DIR //build 发布目录
    },
    plugins: [
      vue(),
      vueJsx(),
      Components({
        resolvers: [ElementPlusResolver()]
      }),
      AutoImport({
        resolvers: [ElementPlusResolver()]
      }),
      topLevelAwait({
        promiseExportName: '__tla',
        promiseImportName: (i) => `__tla_${i}`
      })
    ],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    }
  })
} 