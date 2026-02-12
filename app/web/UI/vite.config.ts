import { fileURLToPath, URL } from 'node:url'

import path, { resolve } from 'path'
import { loadEnv, defineConfig, UserConfig, ConfigEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

import autoImport from 'unplugin-auto-import/vite'
import components from 'unplugin-vue-components/vite' //按需引入
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers' //按需引入 element-plus
import topLevelAwait from 'vite-plugin-top-level-await' //解决 错误 Top-level await is not available

// https://vitejs.dev/config/

const root = process.cwd()
function pathResolve(dir: string) {
  return resolve(root, '.', dir)
}
export default (option: { mode: string }) => {
  process.env = { ...process.env, ...loadEnv(option.mode, process.cwd()) }

  return defineConfig({
    base: process.env.VITE_UI_PATH, // "/audit" ,//, //生成 引用 css 文件和js 文件 增加的前缀
    server: {
      hmr: {
        overlay: false
      }
    },
    build: {
      outDir: process.env.VITE_APP_DIR, //build 发布目录
      rollupOptions: {
        external: ['vue-schart'],
        input: {
          index: path.resolve(__dirname, 'index.html')
          //home: path.resolve(__dirname, 'home0.html'),
          //home: path.resolve(__dirname, 'home.html')
        },
        output: {
          chunkFileNames: 'static/js/[name]-[hash].js',
          entryFileNames: 'static/js/[name]-[hash].js',
          assetFileNames: (assetInfo) => {
            const fileName = assetInfo.name!//|| assetInfo.fileName
            const { name, ext } = path.parse(fileName)
            const imgsExt = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
            if (ext === '.css') {
              return `static/css/${name}-[hash]${ext}`

            } else if (imgsExt.includes(ext)) {
              return `static/img/${name}-[hash]${ext}`
            } else {
              return `static/${ext.replace('.', '')}/${name}-[hash]${ext}`
            }
          }
        }
        /*
        input: {
          app1: 'src/main.ts',
          app2: 'src/mp-main.ts'
        }
          */
      }
    },
    plugins: [
      vue(),
      vueJsx(),
      components({
        resolvers: [ElementPlusResolver()]
      }),
      autoImport({
        resolvers: [ElementPlusResolver()]
      }),
      topLevelAwait({
        promiseExportName: '__tla',
        promiseImportName: (i) => `__tla_${i}`
      })
    ],
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src'),
        vue: 'vue/dist/vue.esm-bundler.js'
      } //npm install --save-dev @types/node
    }
  })
}
