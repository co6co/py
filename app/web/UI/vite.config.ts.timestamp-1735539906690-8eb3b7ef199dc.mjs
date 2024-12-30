// vite.config.ts
import path, { resolve } from "path";
import { loadEnv, defineConfig } from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/vite/dist/node/index.js";
import vue from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/@vitejs/plugin-vue/dist/index.mjs";
import vueJsx from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/@vitejs/plugin-vue-jsx/dist/index.mjs";
import autoImport from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/unplugin-auto-import/dist/vite.js";
import components from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/unplugin-vue-components/dist/vite.js";
import { ElementPlusResolver } from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/unplugin-vue-components/dist/resolvers.js";
import topLevelAwait from "file:///H:/Work/Projects/html/py/app/web/UI/node_modules/vite-plugin-top-level-await/exports/import.mjs";
var __vite_injected_original_dirname = "H:\\Work\\Projects\\html\\py\\app\\web\\UI";
var root = process.cwd();
var vite_config_default = (option) => {
  process.env = { ...process.env, ...loadEnv(option.mode, process.cwd()) };
  return defineConfig({
    base: process.env.VITE_UI_PATH,
    // "/audit" ,//, //生成 引用 css 文件和js 文件 增加的前缀
    server: {
      hmr: {
        overlay: false
      }
    },
    build: {
      outDir: process.env.VITE_APP_DIR,
      //build 发布目录
      rollupOptions: {
        external: ["vue-schart"],
        input: {
          index: path.resolve(__vite_injected_original_dirname, "index.html")
          //home: path.resolve(__dirname, 'home0.html'),
          //home: path.resolve(__dirname, 'home.html')
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
        promiseExportName: "__tla",
        promiseImportName: (i) => `__tla_${i}`
      })
    ],
    resolve: {
      alias: {
        "@": resolve(__vite_injected_original_dirname, "src"),
        vue: "vue/dist/vue.esm-bundler.js"
      }
      //npm install --save-dev @types/node
    }
  });
};
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJIOlxcXFxXb3JrXFxcXFByb2plY3RzXFxcXGh0bWxcXFxccHlcXFxcYXBwXFxcXHdlYlxcXFxVSVwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiSDpcXFxcV29ya1xcXFxQcm9qZWN0c1xcXFxodG1sXFxcXHB5XFxcXGFwcFxcXFx3ZWJcXFxcVUlcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0g6L1dvcmsvUHJvamVjdHMvaHRtbC9weS9hcHAvd2ViL1VJL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IHsgZmlsZVVSTFRvUGF0aCwgVVJMIH0gZnJvbSAnbm9kZTp1cmwnXHJcblxyXG5pbXBvcnQgcGF0aCwgeyByZXNvbHZlIH0gZnJvbSAncGF0aCdcclxuaW1wb3J0IHsgbG9hZEVudiwgZGVmaW5lQ29uZmlnLCBVc2VyQ29uZmlnLCBDb25maWdFbnYgfSBmcm9tICd2aXRlJ1xyXG5pbXBvcnQgdnVlIGZyb20gJ0B2aXRlanMvcGx1Z2luLXZ1ZSdcclxuaW1wb3J0IHZ1ZUpzeCBmcm9tICdAdml0ZWpzL3BsdWdpbi12dWUtanN4J1xyXG5cclxuaW1wb3J0IGF1dG9JbXBvcnQgZnJvbSAndW5wbHVnaW4tYXV0by1pbXBvcnQvdml0ZSdcclxuaW1wb3J0IGNvbXBvbmVudHMgZnJvbSAndW5wbHVnaW4tdnVlLWNvbXBvbmVudHMvdml0ZScgLy9cdTYzMDlcdTk3MDBcdTVGMTVcdTUxNjVcclxuaW1wb3J0IHsgRWxlbWVudFBsdXNSZXNvbHZlciB9IGZyb20gJ3VucGx1Z2luLXZ1ZS1jb21wb25lbnRzL3Jlc29sdmVycycgLy9cdTYzMDlcdTk3MDBcdTVGMTVcdTUxNjUgZWxlbWVudC1wbHVzXHJcbmltcG9ydCB0b3BMZXZlbEF3YWl0IGZyb20gJ3ZpdGUtcGx1Z2luLXRvcC1sZXZlbC1hd2FpdCcgLy9cdTg5RTNcdTUxQjMgXHU5NTE5XHU4QkVGIFRvcC1sZXZlbCBhd2FpdCBpcyBub3QgYXZhaWxhYmxlXHJcblxyXG4vLyBodHRwczovL3ZpdGVqcy5kZXYvY29uZmlnL1xyXG5cclxuY29uc3Qgcm9vdCA9IHByb2Nlc3MuY3dkKClcclxuZnVuY3Rpb24gcGF0aFJlc29sdmUoZGlyOiBzdHJpbmcpIHtcclxuICByZXR1cm4gcmVzb2x2ZShyb290LCAnLicsIGRpcilcclxufVxyXG5leHBvcnQgZGVmYXVsdCAob3B0aW9uOiB7IG1vZGU6IHN0cmluZyB9KSA9PiB7XHJcbiAgcHJvY2Vzcy5lbnYgPSB7IC4uLnByb2Nlc3MuZW52LCAuLi5sb2FkRW52KG9wdGlvbi5tb2RlLCBwcm9jZXNzLmN3ZCgpKSB9XHJcblxyXG4gIHJldHVybiBkZWZpbmVDb25maWcoe1xyXG4gICAgYmFzZTogcHJvY2Vzcy5lbnYuVklURV9VSV9QQVRILCAvLyBcIi9hdWRpdFwiICwvLywgLy9cdTc1MUZcdTYyMTAgXHU1RjE1XHU3NTI4IGNzcyBcdTY1ODdcdTRFRjZcdTU0OENqcyBcdTY1ODdcdTRFRjYgXHU1ODlFXHU1MkEwXHU3Njg0XHU1MjREXHU3RjAwXHJcbiAgICBzZXJ2ZXI6IHtcclxuICAgICAgaG1yOiB7XHJcbiAgICAgICAgb3ZlcmxheTogZmFsc2VcclxuICAgICAgfVxyXG4gICAgfSxcclxuICAgIGJ1aWxkOiB7XHJcbiAgICAgIG91dERpcjogcHJvY2Vzcy5lbnYuVklURV9BUFBfRElSLCAvL2J1aWxkIFx1NTNEMVx1NUUwM1x1NzZFRVx1NUY1NVxyXG4gICAgICByb2xsdXBPcHRpb25zOiB7XHJcbiAgICAgICAgZXh0ZXJuYWw6IFsndnVlLXNjaGFydCddLFxyXG4gICAgICAgIGlucHV0OiB7XHJcbiAgICAgICAgICBpbmRleDogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJ2luZGV4Lmh0bWwnKVxyXG4gICAgICAgICAgLy9ob21lOiBwYXRoLnJlc29sdmUoX19kaXJuYW1lLCAnaG9tZTAuaHRtbCcpLFxyXG4gICAgICAgICAgLy9ob21lOiBwYXRoLnJlc29sdmUoX19kaXJuYW1lLCAnaG9tZS5odG1sJylcclxuICAgICAgICB9XHJcbiAgICAgICAgLypcclxuICAgICAgICBpbnB1dDoge1xyXG4gICAgICAgICAgYXBwMTogJ3NyYy9tYWluLnRzJyxcclxuICAgICAgICAgIGFwcDI6ICdzcmMvbXAtbWFpbi50cydcclxuICAgICAgICB9XHJcbiAgICAgICAgICAqL1xyXG4gICAgICB9XHJcbiAgICB9LFxyXG4gICAgcGx1Z2luczogW1xyXG4gICAgICB2dWUoKSxcclxuICAgICAgdnVlSnN4KCksXHJcbiAgICAgIGNvbXBvbmVudHMoe1xyXG4gICAgICAgIHJlc29sdmVyczogW0VsZW1lbnRQbHVzUmVzb2x2ZXIoKV1cclxuICAgICAgfSksXHJcbiAgICAgIGF1dG9JbXBvcnQoe1xyXG4gICAgICAgIHJlc29sdmVyczogW0VsZW1lbnRQbHVzUmVzb2x2ZXIoKV1cclxuICAgICAgfSksXHJcbiAgICAgIHRvcExldmVsQXdhaXQoe1xyXG4gICAgICAgIHByb21pc2VFeHBvcnROYW1lOiAnX190bGEnLFxyXG4gICAgICAgIHByb21pc2VJbXBvcnROYW1lOiAoaSkgPT4gYF9fdGxhXyR7aX1gXHJcbiAgICAgIH0pXHJcbiAgICBdLFxyXG4gICAgcmVzb2x2ZToge1xyXG4gICAgICBhbGlhczoge1xyXG4gICAgICAgICdAJzogcmVzb2x2ZShfX2Rpcm5hbWUsICdzcmMnKSxcclxuICAgICAgICB2dWU6ICd2dWUvZGlzdC92dWUuZXNtLWJ1bmRsZXIuanMnXHJcbiAgICAgIH0gLy9ucG0gaW5zdGFsbCAtLXNhdmUtZGV2IEB0eXBlcy9ub2RlXHJcbiAgICB9XHJcbiAgfSlcclxufVxyXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBRUEsT0FBTyxRQUFRLGVBQWU7QUFDOUIsU0FBUyxTQUFTLG9CQUEyQztBQUM3RCxPQUFPLFNBQVM7QUFDaEIsT0FBTyxZQUFZO0FBRW5CLE9BQU8sZ0JBQWdCO0FBQ3ZCLE9BQU8sZ0JBQWdCO0FBQ3ZCLFNBQVMsMkJBQTJCO0FBQ3BDLE9BQU8sbUJBQW1CO0FBVjFCLElBQU0sbUNBQW1DO0FBY3pDLElBQU0sT0FBTyxRQUFRLElBQUk7QUFJekIsSUFBTyxzQkFBUSxDQUFDLFdBQTZCO0FBQzNDLFVBQVEsTUFBTSxFQUFFLEdBQUcsUUFBUSxLQUFLLEdBQUcsUUFBUSxPQUFPLE1BQU0sUUFBUSxJQUFJLENBQUMsRUFBRTtBQUV2RSxTQUFPLGFBQWE7QUFBQSxJQUNsQixNQUFNLFFBQVEsSUFBSTtBQUFBO0FBQUEsSUFDbEIsUUFBUTtBQUFBLE1BQ04sS0FBSztBQUFBLFFBQ0gsU0FBUztBQUFBLE1BQ1g7QUFBQSxJQUNGO0FBQUEsSUFDQSxPQUFPO0FBQUEsTUFDTCxRQUFRLFFBQVEsSUFBSTtBQUFBO0FBQUEsTUFDcEIsZUFBZTtBQUFBLFFBQ2IsVUFBVSxDQUFDLFlBQVk7QUFBQSxRQUN2QixPQUFPO0FBQUEsVUFDTCxPQUFPLEtBQUssUUFBUSxrQ0FBVyxZQUFZO0FBQUE7QUFBQTtBQUFBLFFBRzdDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFPRjtBQUFBLElBQ0Y7QUFBQSxJQUNBLFNBQVM7QUFBQSxNQUNQLElBQUk7QUFBQSxNQUNKLE9BQU87QUFBQSxNQUNQLFdBQVc7QUFBQSxRQUNULFdBQVcsQ0FBQyxvQkFBb0IsQ0FBQztBQUFBLE1BQ25DLENBQUM7QUFBQSxNQUNELFdBQVc7QUFBQSxRQUNULFdBQVcsQ0FBQyxvQkFBb0IsQ0FBQztBQUFBLE1BQ25DLENBQUM7QUFBQSxNQUNELGNBQWM7QUFBQSxRQUNaLG1CQUFtQjtBQUFBLFFBQ25CLG1CQUFtQixDQUFDLE1BQU0sU0FBUyxDQUFDO0FBQUEsTUFDdEMsQ0FBQztBQUFBLElBQ0g7QUFBQSxJQUNBLFNBQVM7QUFBQSxNQUNQLE9BQU87QUFBQSxRQUNMLEtBQUssUUFBUSxrQ0FBVyxLQUFLO0FBQUEsUUFDN0IsS0FBSztBQUFBLE1BQ1A7QUFBQTtBQUFBLElBQ0Y7QUFBQSxFQUNGLENBQUM7QUFDSDsiLAogICJuYW1lcyI6IFtdCn0K
