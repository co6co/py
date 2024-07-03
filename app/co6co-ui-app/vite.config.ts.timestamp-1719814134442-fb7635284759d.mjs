// vite.config.ts
import { fileURLToPath, URL } from "node:url";
import { loadEnv, defineConfig } from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/vite/dist/node/index.js";
import vue from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/@vitejs/plugin-vue/dist/index.mjs";
import vueJsx from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/@vitejs/plugin-vue-jsx/dist/index.mjs";
import autoImport from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/unplugin-auto-import/dist/vite.js";
import components from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/unplugin-vue-components/dist/vite.js";
import { ElementPlusResolver } from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/unplugin-vue-components/dist/resolvers.js";
import topLevelAwait from "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/node_modules/vite-plugin-top-level-await/exports/import.mjs";
var __vite_injected_original_import_meta_url = "file:///H:/Work/Projects/html/py/app/co6co-ui-examples/vite.config.ts";
var root = process.cwd();
var vite_config_default = (option) => {
  process.env = { ...process.env, ...loadEnv(option.mode, process.cwd()) };
  return defineConfig({
    base: process.env.VITE_UI_PATH,
    // "/audit" ,//, //生成 引用 css 文件和js 文件 增加的前缀
    build: {
      outDir: process.env.VITE_APP_DIR,
      //build 发布目录
      rollupOptions: {
        external: ["vue-schart", "md-editor-v3", "md-editor-v3/lib/style.css"]
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
        "@": fileURLToPath(new URL("./src", __vite_injected_original_import_meta_url))
      }
    }
  });
};
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJIOlxcXFxXb3JrXFxcXFByb2plY3RzXFxcXGh0bWxcXFxccHlcXFxcYXBwXFxcXGNvNmNvLXVpLWV4YW1wbGVzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJIOlxcXFxXb3JrXFxcXFByb2plY3RzXFxcXGh0bWxcXFxccHlcXFxcYXBwXFxcXGNvNmNvLXVpLWV4YW1wbGVzXFxcXHZpdGUuY29uZmlnLnRzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9IOi9Xb3JrL1Byb2plY3RzL2h0bWwvcHkvYXBwL2NvNmNvLXVpLWV4YW1wbGVzL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IHsgZmlsZVVSTFRvUGF0aCwgVVJMIH0gZnJvbSAnbm9kZTp1cmwnXG5pbXBvcnQgeyByZXNvbHZlIH0gZnJvbSAncGF0aCdcbmltcG9ydCB7IGxvYWRFbnYsIGRlZmluZUNvbmZpZywgVXNlckNvbmZpZywgQ29uZmlnRW52IH0gZnJvbSAndml0ZSdcbmltcG9ydCB2dWUgZnJvbSAnQHZpdGVqcy9wbHVnaW4tdnVlJ1xuaW1wb3J0IHZ1ZUpzeCBmcm9tICdAdml0ZWpzL3BsdWdpbi12dWUtanN4J1xuXG5pbXBvcnQgYXV0b0ltcG9ydCBmcm9tICd1bnBsdWdpbi1hdXRvLWltcG9ydC92aXRlJ1xuaW1wb3J0IGNvbXBvbmVudHMgZnJvbSAndW5wbHVnaW4tdnVlLWNvbXBvbmVudHMvdml0ZScgLy9cdTYzMDlcdTk3MDBcdTVGMTVcdTUxNjVcbmltcG9ydCB7IEVsZW1lbnRQbHVzUmVzb2x2ZXIgfSBmcm9tICd1bnBsdWdpbi12dWUtY29tcG9uZW50cy9yZXNvbHZlcnMnIC8vXHU2MzA5XHU5NzAwXHU1RjE1XHU1MTY1IGVsZW1lbnQtcGx1c1xuaW1wb3J0IHRvcExldmVsQXdhaXQgZnJvbSAndml0ZS1wbHVnaW4tdG9wLWxldmVsLWF3YWl0JyAvL1x1ODlFM1x1NTFCMyBcdTk1MTlcdThCRUYgVG9wLWxldmVsIGF3YWl0IGlzIG5vdCBhdmFpbGFibGVcblxuLy8gaHR0cHM6Ly92aXRlanMuZGV2L2NvbmZpZy9cblxuY29uc3Qgcm9vdCA9IHByb2Nlc3MuY3dkKClcbmZ1bmN0aW9uIHBhdGhSZXNvbHZlKGRpcjogc3RyaW5nKSB7XG4gIHJldHVybiByZXNvbHZlKHJvb3QsICcuJywgZGlyKVxufVxuZXhwb3J0IGRlZmF1bHQgKG9wdGlvbjogeyBtb2RlOiBzdHJpbmcgfSkgPT4ge1xuICBwcm9jZXNzLmVudiA9IHsgLi4ucHJvY2Vzcy5lbnYsIC4uLmxvYWRFbnYob3B0aW9uLm1vZGUsIHByb2Nlc3MuY3dkKCkpIH1cblxuICByZXR1cm4gZGVmaW5lQ29uZmlnKHtcbiAgICBiYXNlOiBwcm9jZXNzLmVudi5WSVRFX1VJX1BBVEgsIC8vIFwiL2F1ZGl0XCIgLC8vLCAvL1x1NzUxRlx1NjIxMCBcdTVGMTVcdTc1MjggY3NzIFx1NjU4N1x1NEVGNlx1NTQ4Q2pzIFx1NjU4N1x1NEVGNiBcdTU4OUVcdTUyQTBcdTc2ODRcdTUyNERcdTdGMDBcbiAgICBidWlsZDoge1xuICAgICAgb3V0RGlyOiBwcm9jZXNzLmVudi5WSVRFX0FQUF9ESVIsIC8vYnVpbGQgXHU1M0QxXHU1RTAzXHU3NkVFXHU1RjU1XG4gICAgICByb2xsdXBPcHRpb25zOiB7XG4gICAgICAgIGV4dGVybmFsOiBbJ3Z1ZS1zY2hhcnQnLCAnbWQtZWRpdG9yLXYzJywgJ21kLWVkaXRvci12My9saWIvc3R5bGUuY3NzJ10sXG4gICAgICB9LFxuICAgIH0sXG4gICAgcGx1Z2luczogW1xuICAgICAgdnVlKCksXG4gICAgICB2dWVKc3goKSxcbiAgICAgIGNvbXBvbmVudHMoe1xuICAgICAgICByZXNvbHZlcnM6IFtFbGVtZW50UGx1c1Jlc29sdmVyKCldLFxuICAgICAgfSksXG4gICAgICBhdXRvSW1wb3J0KHtcbiAgICAgICAgcmVzb2x2ZXJzOiBbRWxlbWVudFBsdXNSZXNvbHZlcigpXSxcbiAgICAgIH0pLFxuICAgICAgdG9wTGV2ZWxBd2FpdCh7XG4gICAgICAgIHByb21pc2VFeHBvcnROYW1lOiAnX190bGEnLFxuICAgICAgICBwcm9taXNlSW1wb3J0TmFtZTogKGkpID0+IGBfX3RsYV8ke2l9YCxcbiAgICAgIH0pLFxuICAgIF0sXG4gICAgcmVzb2x2ZToge1xuICAgICAgYWxpYXM6IHtcbiAgICAgICAgJ0AnOiBmaWxlVVJMVG9QYXRoKG5ldyBVUkwoJy4vc3JjJywgaW1wb3J0Lm1ldGEudXJsKSksXG4gICAgICB9LFxuICAgIH0sXG4gIH0pXG59XG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQTBVLFNBQVMsZUFBZSxXQUFXO0FBRTdXLFNBQVMsU0FBUyxvQkFBMkM7QUFDN0QsT0FBTyxTQUFTO0FBQ2hCLE9BQU8sWUFBWTtBQUVuQixPQUFPLGdCQUFnQjtBQUN2QixPQUFPLGdCQUFnQjtBQUN2QixTQUFTLDJCQUEyQjtBQUNwQyxPQUFPLG1CQUFtQjtBQVR1TCxJQUFNLDJDQUEyQztBQWFsUSxJQUFNLE9BQU8sUUFBUSxJQUFJO0FBSXpCLElBQU8sc0JBQVEsQ0FBQyxXQUE2QjtBQUMzQyxVQUFRLE1BQU0sRUFBRSxHQUFHLFFBQVEsS0FBSyxHQUFHLFFBQVEsT0FBTyxNQUFNLFFBQVEsSUFBSSxDQUFDLEVBQUU7QUFFdkUsU0FBTyxhQUFhO0FBQUEsSUFDbEIsTUFBTSxRQUFRLElBQUk7QUFBQTtBQUFBLElBQ2xCLE9BQU87QUFBQSxNQUNMLFFBQVEsUUFBUSxJQUFJO0FBQUE7QUFBQSxNQUNwQixlQUFlO0FBQUEsUUFDYixVQUFVLENBQUMsY0FBYyxnQkFBZ0IsNEJBQTRCO0FBQUEsTUFDdkU7QUFBQSxJQUNGO0FBQUEsSUFDQSxTQUFTO0FBQUEsTUFDUCxJQUFJO0FBQUEsTUFDSixPQUFPO0FBQUEsTUFDUCxXQUFXO0FBQUEsUUFDVCxXQUFXLENBQUMsb0JBQW9CLENBQUM7QUFBQSxNQUNuQyxDQUFDO0FBQUEsTUFDRCxXQUFXO0FBQUEsUUFDVCxXQUFXLENBQUMsb0JBQW9CLENBQUM7QUFBQSxNQUNuQyxDQUFDO0FBQUEsTUFDRCxjQUFjO0FBQUEsUUFDWixtQkFBbUI7QUFBQSxRQUNuQixtQkFBbUIsQ0FBQyxNQUFNLFNBQVMsQ0FBQztBQUFBLE1BQ3RDLENBQUM7QUFBQSxJQUNIO0FBQUEsSUFDQSxTQUFTO0FBQUEsTUFDUCxPQUFPO0FBQUEsUUFDTCxLQUFLLGNBQWMsSUFBSSxJQUFJLFNBQVMsd0NBQWUsQ0FBQztBQUFBLE1BQ3REO0FBQUEsSUFDRjtBQUFBLEVBQ0YsQ0FBQztBQUNIOyIsCiAgIm5hbWVzIjogW10KfQo=
